# 第九章。人员检测：构建一个应用程序

如果你问人们哪种感官对他们的日常生活影响最大，很多人会回答视觉。

视觉是一种极其有用的感觉。它使无数自然生物能够在环境中导航，找到食物来源，并避免遇到危险。作为人类，视觉帮助我们认识朋友，解释象征性信息，并理解我们周围的世界，而无需过于接近。

直到最近，视觉的力量并不可用于机器。我们大多数的机器人只是用触摸和接近传感器在世界中探索，通过一系列碰撞获取其结构的知识。一眼之间，一个人可以向你描述一个物体的形状、属性和目的，而无需与之互动。机器人就没有这样的运气。视觉信息只是太混乱、无结构和难以解释了。

随着卷积神经网络的发展，构建能够“看到”的程序变得容易。受到哺乳动物视觉皮层结构的启发，CNN学会了理解我们的视觉世界，将一个极其复杂的输入过滤成已知模式和形状的地图。这些部分的精确组合可以告诉我们在给定数字图像中存在的实体。

如今，视觉模型被用于许多不同的任务。自动驾驶车辆使用视觉来发现道路上的危险。工厂机器人使用摄像头捕捉有缺陷的零件。研究人员已经训练出可以从医学图像中诊断疾病的模型。而且你的智能手机很有可能在照片中识别出人脸，以确保它们焦点完美。

具有视觉的机器可以帮助改变我们的家庭和城市，自动化以前无法实现的家务。但视觉是一种亲密的感觉。我们大多数人不喜欢自己的行为被记录，或者我们的生活被实时传输到云端，这通常是ML推断的地方。

想象一下一个可以通过内置摄像头“看到”的家用电器。它可以是一个可以发现入侵者的安全系统，一个知道自己被遗弃的炉灶，或者一个在房间里没有人时自动关闭的电视。在这些情况下，隐私至关重要。即使没有人观看录像，互联网连接的摄像头嵌入在始终开启的设备中的安全隐患使它们对大多数消费者不吸引人。

但所有这些都随着TinyML而改变。想象一下一个智能炉灶，如果长时间不被注意就会关闭它的燃烧器。如果它可以“看到”附近有一个使用微型微控制器的厨师，而没有任何与互联网的连接，我们就可以获得智能设备的所有好处，而不会有任何隐私方面的妥协。

更重要的是，具有视觉功能的微型设备可以进入以前没有敢去的地方。基于微控制器的视觉系统由于其微小的功耗，可以在一个小电池上运行数月甚至数年。这些设备可以在丛林或珊瑚礁中计算濒危动物的数量，而无需在线。

同样的技术使得构建一个视觉传感器作为一个独立的电子组件成为可能。传感器输出1表示某个物体在视野中，输出0表示不在视野中，但它从不分享摄像头收集的任何图像数据。这种类型的传感器可以嵌入各种产品中，从智能家居系统到个人车辆。你的自行车可以在你后面有车时闪光灯。你的空调可以知道有人在家。而且因为图像数据从未离开独立的传感器，即使产品连接到互联网，也可以保证安全。

本章探讨的应用程序使用一个预训练的人体检测模型，在连接了摄像头的微控制器上运行，以知道何时有人在视野中。在[第10章](ch10.xhtml#chapter_person_detection_training)中，您将了解这个模型是如何工作的，以及如何训练自己的模型来检测您想要的内容。

阅读完本章后，您将了解如何在微控制器上处理摄像头数据，以及如何使用视觉模型运行推断并解释输出。您可能会惊讶于这实际上是多么容易！

# 我们正在构建什么

我们将构建一个嵌入式应用程序，该应用程序使用模型对摄像头捕获的图像进行分类。该模型经过训练，能够识别摄像头输入中是否存在人物。这意味着我们的应用程序将能够检测人物的存在或缺席，并相应地产生输出。

这本质上是我们稍早描述的智能视觉传感器。当检测到人物时，我们的示例代码将点亮LED灯—但您可以扩展它以控制各种项目。

###### 注意

与我们在[第7章](ch07.xhtml#chapter_speech_wake_word_example)中开发的应用程序一样，您可以在[TensorFlow GitHub存储库](https://oreil.ly/9aLhs)中找到此应用程序的源代码。

与之前的章节一样，我们首先浏览测试和应用程序代码，然后是使示例在各种设备上运行的逻辑。

我们提供了将应用程序部署到以下微控制器平台的说明：

+   [Arduino Nano 33 BLE Sense](https://oreil.ly/6qlMD)

+   [SparkFun Edge](https://oreil.ly/-hoL-)

###### 注意

TensorFlow Lite定期添加对新设备的支持，因此如果您想要使用的设备未在此处列出，请查看示例的[*README.md*](https://oreil.ly/6gRlo)。如果在按照这些步骤操作时遇到问题，您也可以在那里查找更新的部署说明。

与之前的章节不同，您需要一些额外的硬件来运行这个应用程序。因为这两个开发板都没有集成摄像头，我们建议购买一个*摄像头模块*。您将在每个设备的部分中找到这些信息。

让我们从了解应用程序的结构开始。它比您想象的要简单得多。

# 应用程序架构

到目前为止，我们已经确定了嵌入式机器学习应用程序执行以下一系列操作：

1.  获取输入。

1.  对输入进行预处理，提取适合输入模型的特征。

1.  对处理后的输入运行推断。

1.  对模型的输出进行后处理以理解其含义。

1.  使用得到的信息来实现所需的功能。

在[第7章](ch07.xhtml#chapter_speech_wake_word_example)中，我们看到这种方法应用于唤醒词检测，其输入是音频。这一次，我们的输入将是图像数据。这听起来可能更复杂，但实际上比音频更容易处理。

图像数据通常表示为像素值数组。我们将从嵌入式摄像头模块获取图像数据，所有这些模块都以这种格式提供数据。我们的模型也期望其输入是像素值数组。因此，在将数据输入模型之前，我们不需要进行太多的预处理。

鉴于我们不需要进行太多的预处理，我们的应用程序将会相当简单。它从摄像头中获取数据快照，将其输入模型，并确定检测到了哪个输出类。然后以一种简单的方式显示结果。

在我们继续之前，让我们更多地了解一下我们将要使用的模型。

## 介绍我们的模型

在[第7章](ch07.xhtml#chapter_speech_wake_word_example)中，我们了解到卷积神经网络是专门设计用于处理多维张量的神经网络，其中信息包含在相邻值组之间的关系中。它们特别适合处理图像数据。

我们的人体检测模型是一个卷积神经网络，训练于[Visual Wake Words数据集](https://oreil.ly/EC6nd)。该数据集包含115,000张图像，每张图像都标记了是否包含人体。

该模型大小为250 KB，比我们的语音模型大得多。除了占用更多内存外，这种额外的大小意味着运行单个推断需要更长的时间。

该模型接受96×96像素的灰度图像作为输入。每个图像都以形状为`(96, 96, 1)`的3D张量提供，其中最后一个维度包含一个表示单个像素的8位值。该值指定像素的阴影，范围从0（完全黑色）到255（完全白色）。

我们的摄像头模块可以以各种分辨率返回图像，因此我们需要确保它们被调整为96×96像素。我们还需要将全彩图像转换为灰度图像，以便与模型配合使用。

您可能认为96×96像素听起来像是一个很小的分辨率，但它将足以让我们在每个图像中检测到一个人。处理图像的模型通常接受令人惊讶地小的分辨率。增加模型的输入尺寸会带来递减的回报，而网络的复杂性会随着输入规模的增加而大幅增加。因此，即使是最先进的图像分类模型通常也只能处理最大为320×320像素的图像。

模型输出两个概率：一个指示输入中是否存在人的概率，另一个指示是否没有人的概率。概率范围从0到255。

我们的人体检测模型使用了*MobileNet*架构，这是一个为移动手机等设备设计的用于图像分类的经过广泛测试的架构。在[第10章](ch10.xhtml#chapter_person_detection_training)中，您将学习如何将该模型适配到微控制器上，并且如何训练您自己的模型。现在，让我们继续探索我们的应用程序是如何工作的。

## 所有的组件

[图9-1](#application_architecture_2)显示了我们人体检测应用程序的结构。

![我们人体检测应用程序的组件图示](Images/timl_0901.png)

###### 图9-1。我们人体检测应用程序的组件

正如我们之前提到的，这比唤醒词应用程序要简单得多，因为我们可以直接将图像数据传递到模型中，无需预处理。

另一个让事情简单的方面是我们不对模型的输出进行平均。我们的唤醒词模型每秒运行多次，因此我们必须对其输出进行平均以获得稳定的结果。我们的人体检测模型更大，推断时间更长。这意味着不需要对其输出进行平均。

代码有五个主要部分：

主循环

与其他示例一样，我们的应用程序在一个连续循环中运行。然而，由于我们的模型更大更复杂，因此推断的运行时间会更长。根据设备的不同，我们可以预期每隔几秒进行一次推断，而不是每秒进行多次推断。

图像提供者

该组件从摄像头捕获图像数据并将其写入输入张量。捕获图像的方法因设备而异，因此该组件可以被覆盖和自定义。

TensorFlow Lite解释器

解释器运行TensorFlow Lite模型，将输入图像转换为一组概率。

模型

该模型作为数据数组包含在内，并由解释器运行。250 KB的模型太大了，无法提交到TensorFlow GitHub存储库。因此，在构建项目时，Makefile会下载它。如果您想查看，可以自行下载[*tf_lite_micro_person_data_grayscale.zip*](https://oreil.ly/Ylq9m)。

检测响应器

检测响应器接收模型输出的概率，并使用设备的输出功能来显示它们。我们可以为不同的设备类型进行覆盖。在我们的示例代码中，它将点亮LED，但您可以扩展它以执行几乎任何操作。

为了了解这些部分如何配合，我们将查看它们的测试。

# 通过测试

这个应用程序非常简单，因为只有几个测试需要进行。您可以在[GitHub存储库](https://oreil.ly/31vB5)中找到它们：

[*person_detection_test.cc*](https://oreil.ly/r4ny8)

展示如何对表示单个图像的数组运行推断

[*image_provider_test.cc*](https://oreil.ly/Js6M3)

展示如何使用图像提供程序捕获图像

[*detection_responder_test.cc*](https://oreil.ly/KBVLF)

展示如何使用检测响应器输出检测结果

让我们从探索*person_detection_test.cc*开始，看看如何对图像数据运行推断。因为这是我们走过的第三个示例，这段代码应该感觉相当熟悉。您已经在成为嵌入式ML开发人员的道路上取得了很大进展！

## 基本流程

首先是*person_detection_test.cc*。我们首先引入模型需要的操作：

```py
namespace tflite {
namespace ops {
namespace micro {
TfLiteRegistration* Register_DEPTHWISE_CONV_2D();
TfLiteRegistration* Register_CONV_2D();
TfLiteRegistration* Register_AVERAGE_POOL_2D();
}  // namespace micro
}  // namespace ops
}  // namespace tflite
```

接下来，我们定义一个适合模型大小的张量区域。通常情况下，这个数字是通过试错确定的：

```py
const int tensor_arena_size = 70 * 1024;
uint8_t tensor_arena[tensor_arena_size];
```

然后我们进行典型的设置工作，准备解释器运行，包括使用`MicroMutableOpResolver`注册必要的操作：

```py
// Set up logging.
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

// Map the model into a usable data structure. This doesn't involve any
// copying or parsing, it's a very lightweight operation.
const tflite::Model* model = ::tflite::GetModel(g_person_detect_model_data);
if (model->version() != TFLITE_SCHEMA_VERSION) {
error_reporter->Report(
    "Model provided is schema version %d not equal "
    "to supported version %d.\n",
    model->version(), TFLITE_SCHEMA_VERSION);
}

// Pull in only the operation implementations we need.
tflite::MicroMutableOpResolver micro_mutable_op_resolver;
micro_mutable_op_resolver.AddBuiltin(
    tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
    tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                                     tflite::ops::micro::Register_CONV_2D());
micro_mutable_op_resolver.AddBuiltin(
    tflite::BuiltinOperator_AVERAGE_POOL_2D,
    tflite::ops::micro::Register_AVERAGE_POOL_2D());

// Build an interpreter to run the model with.
tflite::MicroInterpreter interpreter(model, micro_mutable_op_resolver,
                                     tensor_arena, tensor_arena_size,
                                     error_reporter);
interpreter.AllocateTensors();
```

我们的下一步是检查输入张量。我们检查它是否具有预期数量的维度，以及其维度是否适当：

```py
// Get information about the memory area to use for the model's input.
TfLiteTensor* input = interpreter.input(0);

// Make sure the input has the properties we expect.
TF_LITE_MICRO_EXPECT_NE(nullptr, input);
TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
TF_LITE_MICRO_EXPECT_EQ(kNumRows, input->dims->data[1]);
TF_LITE_MICRO_EXPECT_EQ(kNumCols, input->dims->data[2]);
TF_LITE_MICRO_EXPECT_EQ(kNumChannels, input->dims->data[3]);
TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, input->type);
```

从中我们可以看到，输入技术上是一个5D张量。第一个维度只是包含一个元素的包装器。接下来的两个维度表示图像像素的行和列。最后一个维度保存用于表示每个像素的颜色通道的数量。

告诉我们预期维度的常量`kNumRows`、`kNumCols`和`kNumChannels`在[*model_settings.h*](https://oreil.ly/ae2OI)中定义。它们看起来像这样：

```py
constexpr int kNumCols = 96;
constexpr int kNumRows = 96;
constexpr int kNumChannels = 1;
```

如您所见，模型预计接受一个96×96像素的位图。图像将是灰度的，每个像素有一个颜色通道。

接下来在代码中，我们使用简单的`for`循环将测试图像复制到输入张量中：

```py
// Copy an image with a person into the memory area used for the input.
const uint8_t* person_data = g_person_data;
for (int i = 0; i < input->bytes; ++i) {
    input->data.uint8[i] = person_data[i];
}
```

存储图像数据的变量`g_person_data`由*person_image_data.h*定义。为了避免向存储库添加更多大文件，数据本身会在首次运行测试时作为*tf_lite_micro_person_data_grayscale.zip*的一部分与模型一起下载。

在我们填充了输入张量之后，我们运行推断。这和以往一样简单：

```py
// Run the model on this input and make sure it succeeds.
TfLiteStatus invoke_status = interpreter.Invoke();
if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed\n");
}
TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);
```

现在我们检查输出张量，确保它具有预期的大小和形状：

```py
TfLiteTensor* output = interpreter.output(0);
TF_LITE_MICRO_EXPECT_EQ(4, output->dims->size);
TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[1]);
TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[2]);
TF_LITE_MICRO_EXPECT_EQ(kCategoryCount, output->dims->data[3]);
TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, output->type);
```

模型的输出有四个维度。前三个只是包装器，围绕第四个维度，其中包含模型训练的每个类别的一个元素。

类别的总数作为常量`kCategoryCount`可用，它位于*model_settings.h*中，还有一些其他有用的值：

```py
constexpr int kCategoryCount = 3;
constexpr int kPersonIndex = 1;
constexpr int kNotAPersonIndex = 2;
extern const char* kCategoryLabels[kCategoryCount];
```

正如`kCategoryCount`所示，输出中有三个类别。第一个恰好是一个未使用的类别，我们可以忽略。“人”类别排在第二位，我们可以从常量`kPersonIndex`中存储的索引中看到。“不是人”类别排在第三位，其索引由`kNotAPersonIndex`显示。

还有一个类别标签数组`kCategoryLabels`，在[*model_settings.cc*](https://oreil.ly/AB0zS)中实现：

```py
const char* kCategoryLabels[kCategoryCount] = {
    "unused",
    "person",
    "notperson",
};
```

接下来的代码块记录“人”和“非人”分数，并断言“人”分数更高——因为我们传入的是一个人的图像：

```py
uint8_t person_score = output->data.uint8[kPersonIndex];
uint8_t no_person_score = output->data.uint8[kNotAPersonIndex];
error_reporter->Report(
    "person data.  person score: %d, no person score: %d\n", person_score,
    no_person_score);
TF_LITE_MICRO_EXPECT_GT(person_score, no_person_score);
```

由于输出张量的唯一数据内容是表示类别分数的三个`uint8`值，第一个值未使用，我们可以通过`output->data.uint8[kPersonIndex]`和`output->data.uint8[kNotAPersonIndex]`直接访问分数。作为`uint8`类型，它们的最小值为0，最大值为255。

###### 注意

如果“人”和“非人”分数相似，这可能意味着模型对其预测不太有信心。在这种情况下，您可能选择考虑结果不确定。

接下来，我们测试没有人的图像，由`g_no_person_data`持有：

```py
const uint8_t* no_person_data = g_no_person_data;
for (int i = 0; i < input->bytes; ++i) {
    input->data.uint8[i] = no_person_data[i];
}
```

推理运行后，我们断言“非人”分数更高：

```py
person_score = output->data.uint8[kPersonIndex];
no_person_score = output->data.uint8[kNotAPersonIndex];
error_reporter->Report(
    "no person data.  person score: %d, no person score: %d\n", person_score,
    no_person_score);
TF_LITE_MICRO_EXPECT_GT(no_person_score, person_score);
```

正如您所看到的，这里没有什么花哨的东西。我们可能正在输入图像而不是标量或频谱图，但推理过程与我们以前看到的类似。

运行测试同样简单。只需从TensorFlow存储库的根目录发出以下命令：

```py
make -f tensorflow/lite/micro/tools/make/Makefile \
  test_person_detection_test
```

第一次运行测试时，将下载模型和图像数据。如果您想查看已下载的文件，可以在*tensorflow/lite/micro/tools/make/downloads/person_model_grayscale*中找到它们。

接下来，我们检查图像提供程序的接口。

## 图像提供程序

图像提供程序负责从摄像头获取数据，并以适合写入模型输入张量的格式返回数据。文件[*image_provider.h*](https://oreil.ly/5Vjbe)定义了其接口：

```py
TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
                      int image_height, int channels, uint8_t* image_data);
```

由于其实际实现是特定于平台的，因此在[*person_detection/image_provider.cc*](https://oreil.ly/QoQ3O)中有一个返回虚拟数据的参考实现。

[*image_provider_test.cc*](https://oreil.ly/Nbl9x)中的测试调用此参考实现以展示其用法。我们的首要任务是创建一个数组来保存图像数据。这发生在以下行中：

```py
uint8_t image_data[kMaxImageSize];
```

常量`kMaxImageSize`来自我们的老朋友[*model_settings.h*](https://oreil.ly/5naFK)。

设置了这个数组后，我们可以调用`GetImage()`函数从摄像头捕获图像：

```py
TfLiteStatus get_status =
    GetImage(error_reporter, kNumCols, kNumRows, kNumChannels, image_data);
TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, get_status);
TF_LITE_MICRO_EXPECT_NE(image_data, nullptr);
```

我们使用`ErrorReporter`实例、我们想要的列数、行数和通道数以及指向我们的`image_data`数组的指针来调用它。该函数将把图像数据写入此数组。我们可以检查函数的返回值来确定捕获过程是否成功；如果有问题，它将设置为`kTfLiteError`，否则为`kTfLiteOk`。

最后，测试通过返回的数据以显示所有内存位置都是可读的。即使图像在技术上具有行、列和通道，但实际上数据被展平为一维数组：

```py
uint32_t total = 0;
for (int i = 0; i < kMaxImageSize; ++i) {
    total += image_data[i];
}
```

要运行此测试，请使用以下命令：

```py
make -f tensorflow/lite/micro/tools/make/Makefile \
  test_image_provider_test
```

我们将在本章后面查看*image_provider.cc*的特定于设备的实现；现在，让我们看一下检测响应器的接口。

## 检测响应器

我们的最终测试展示了检测响应器的使用方式。这是负责传达推理结果的代码。其接口在[*detection_responder.h*](https://oreil.ly/cTptj)中定义，测试在[*detection_responder_test.cc*](https://oreil.ly/Igx7a)中。

接口非常简单：

```py
void RespondToDetection(tflite::ErrorReporter* error_reporter,
                        uint8_t person_score, uint8_t no_person_score);
```

我们只需使用“人”和“非人”类别的分数调用它，它将根据情况决定要做什么。

[*detection_responder.cc*](https://oreil.ly/5Wjjt)中的参考实现只是记录这些值。*detection_responder_test.cc*中的测试调用该函数几次：

```py
RespondToDetection(error_reporter, 100, 200);
RespondToDetection(error_reporter, 200, 100);
```

要运行测试并查看输出，请使用以下命令：

```py
make -f tensorflow/lite/micro/tools/make/Makefile \
  test_detection_responder_test
```

我们已经探索了所有测试和它们所练习的接口。现在让我们走一遍程序本身。

# 检测人员

应用程序的核心功能位于[*main_functions.cc*](https://oreil.ly/64oHW)中。它们简短而简洁，我们在测试中已经看到了它们的大部分逻辑。

首先，我们引入模型所需的所有操作：

```py
namespace tflite {
namespace ops {
namespace micro {
TfLiteRegistration* Register_DEPTHWISE_CONV_2D();
TfLiteRegistration* Register_CONV_2D();
TfLiteRegistration* Register_AVERAGE_POOL_2D();
}  // namespace micro
}  // namespace ops
}  // namespace tflite
```

接下来，我们声明一堆变量来保存重要的移动部件：

```py
tflite::ErrorReporter* g_error_reporter = nullptr;
const tflite::Model* g_model = nullptr;
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
```

之后，我们为张量操作分配一些工作内存：

```py
constexpr int g_tensor_arena_size = 70 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
```

在`setup()`函数中，在任何其他操作发生之前运行，我们创建一个错误报告器，加载我们的模型，设置一个解释器实例，并获取模型输入张量的引用：

```py
void setup() {
  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  g_error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  g_model = tflite::GetModel(g_person_detect_model_data);
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    g_error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        g_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  static tflite::MicroMutableOpResolver micro_mutable_op_resolver;
  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                                       tflite::ops::micro::Register_CONV_2D());
  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_AVERAGE_POOL_2D,
      tflite::ops::micro::Register_AVERAGE_POOL_2D());

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_mutable_op_resolver, tensor_arena, kTensorArenaSize,
      error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
}
```

代码的下一部分在程序的主循环中被不断调用。它首先使用图像提供程序获取图像，通过传递一个输入张量的引用，使图像直接写入其中：

```py
void loop() {
  // Get image from provider.
  if (kTfLiteOk != GetImage(g_error_reporter, kNumCols, kNumRows, kNumChannels,
                            g_input->data.uint8)) {
    g_error_reporter->Report("Image capture failed.");
  }
```

然后运行推理，获取输出张量，并从中读取“人”和“无人”分数。这些分数被传递到检测响应器的`RespondToDetection()`函数中：

```py
  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != g_interpreter->Invoke()) {
    g_error_reporter->Report("Invoke failed.");
  }

  TfLiteTensor* output = g_interpreter->output(0);

  // Process the inference results.
  uint8_t person_score = output->data.uint8[kPersonIndex];
  uint8_t no_person_score = output->data.uint8[kNotAPersonIndex];
  RespondToDetection(g_error_reporter, person_score, no_person_score);
}
```

在`RespondToDetection()`完成输出结果后，`loop()`函数将返回，准备好被程序的主循环再次调用。

循环本身在程序的`main()`函数中定义，该函数位于[*main.cc*](https://oreil.ly/_PR3L)中。它一次调用`setup()`函数，然后重复调用`loop()`函数，直到无限循环：

```py
int main(int argc, char* argv[]) {
  setup();
  while (true) {
    loop();
  }
}
```

这就是整个程序！这个例子很棒，因为它表明与复杂的机器学习模型一起工作可以出奇地简单。模型包含了所有的复杂性，我们只需要提供数据给它。

在我们继续之前，您可以在本地运行程序进行尝试。图像提供程序的参考实现只返回虚拟数据，因此您不会得到有意义的识别结果，但至少可以看到代码在运行。

首先，使用以下命令构建程序：

```py
make -f tensorflow/lite/micro/tools/make/Makefile person_detection
```

构建完成后，您可以使用以下命令运行示例：

```py
tensorflow/lite/micro/tools/make/gen/osx_x86_64/bin/ \
person_detection
```

您会看到程序的输出在屏幕上滚动，直到按下Ctrl-C终止它：

```py
person score:129 no person score 202
person score:129 no person score 202
person score:129 no person score 202
person score:129 no person score 202
person score:129 no person score 202
person score:129 no person score 202
```

在接下来的部分中，我们将详细介绍特定设备的代码，该代码将捕获摄像头图像并在每个平台上输出结果。我们还展示了如何部署和运行此代码。

# 部署到微控制器

在这一部分中，我们将代码部署到两个熟悉的设备上：

+   [Arduino Nano 33 BLE Sense](https://oreil.ly/6qlMD)

+   [SparkFun Edge](https://oreil.ly/-hoL-)

这次有一个很大的不同：因为这两个设备都没有内置摄像头，我们建议您为您使用的任何设备购买摄像头模块。每个设备都有自己的*image_provider.cc*实现，它与摄像头模块进行接口，以捕获图像。*detection_responder.cc*中还有特定于设备的输出代码。

这很简单，所以它将是一个很好的模板，用来创建你自己的基于视觉的ML应用程序。

让我们开始探索Arduino的实现。

## Arduino

作为Arduino板，Arduino Nano 33 BLE Sense可以访问大量兼容的第三方硬件和库的生态系统。我们使用了一个专为与Arduino配合使用而设计的第三方摄像头模块，以及一些Arduino库，这些库将与我们的摄像头模块进行接口，并理解其输出的数据。

### 要购买哪种摄像头模块

这个例子使用[Arducam Mini 2MP Plus](https://oreil.ly/LAwhb)摄像头模块。它很容易连接到Arduino Nano 33 BLE Sense，并且可以由Arduino板的电源供应提供电力。它有一个大镜头，能够捕获高质量的200万像素图像 - 尽管我们将使用其内置的图像重缩放功能来获得较小的分辨率。它并不特别节能，但其高质量的图像使其非常适合构建图像捕获应用程序，比如用于记录野生动物。

### 在Arduino上捕获图像

我们通过一些引脚将Arducam模块连接到Arduino板。为了获取图像数据，我们从Arduino板向Arducam发送一个命令，指示它捕获图像。Arducam将执行此操作，将图像存储在其内部数据缓冲区中。然后，我们发送进一步的命令，允许我们从Arducam的内部缓冲区中读取图像数据并将其存储在Arduino的内存中。为了执行所有这些操作，我们使用官方的Arducam库。

Arducam相机模块具有一颗200万像素的图像传感器，分辨率为1920×1080。我们的人体检测模型的输入尺寸仅为96×96，因此我们不需要所有这些数据。事实上，Arduino本身没有足够的内存来容纳一张200万像素的图像，其大小将达到几兆字节。

幸运的是，Arducam硬件具有将输出调整为更小分辨率的能力，即160×120像素。我们可以通过在代码中仅保留中心的96×96像素来轻松将其裁剪为96×96。然而，为了复杂化问题，Arducam的调整大小输出使用了JPEG，这是一种常见的图像压缩格式。我们的模型需要一个像素数组，而不是一个JPEG编码的图像，这意味着我们需要在使用之前解码Arducam的输出。我们可以使用一个开源库来实现这一点。

我们的最后任务是将Arducam的彩色图像输出转换为灰度，这是我们的人体检测模型所期望的。我们将灰度数据写入我们模型的输入张量。

图像提供程序实现在[*arduino/image_provider.cc*](https://oreil.ly/kGx0-)中。我们不会解释其每个细节，因为代码是特定于Arducam相机模块的。相反，让我们以高层次的方式来看一下发生了什么。

`GetImage()`函数是图像提供程序与外部世界的接口。在我们的应用程序主循环中调用它以获取一帧图像数据。第一次调用时，我们需要初始化相机。这通过调用`InitCamera()`函数来实现，如下所示：

```py
  static bool g_is_camera_initialized = false;
  if (!g_is_camera_initialized) {
    TfLiteStatus init_status = InitCamera(error_reporter);
    if (init_status != kTfLiteOk) {
      error_reporter->Report("InitCamera failed");
      return init_status;
    }
    g_is_camera_initialized = true;
  }
```

`InitCamera()`函数在*image_provider.cc*中进一步定义。我们不会在这里详细介绍它，因为它非常特定于设备，如果您想在自己的代码中使用它，只需复制粘贴即可。它配置Arduino的硬件以与Arducam通信，然后确认通信正常工作。最后，它指示Arducam输出160×120像素的JPEG图像。

`GetImage()`函数调用的下一个函数是`PerformCapture()`：

```py
TfLiteStatus capture_status = PerformCapture(error_reporter);
```

我们也不会详细介绍这个函数。它只是向相机模块发送一个命令，指示其捕获图像并将图像数据存储在其内部缓冲区中。然后，它等待确认图像已被捕获。此时，Arducam的内部缓冲区中有图像数据，但Arduino本身还没有任何图像数据。

接下来我们调用的函数是`ReadData()`：

```py
  TfLiteStatus read_data_status = ReadData(error_reporter);
```

`ReadData()`函数使用更多的命令从Arducam获取图像数据。函数运行后，全局变量`jpeg_buffer`将填充从相机检索到的JPEG编码图像数据。

当我们有JPEG编码的图像时，我们的下一步是将其解码为原始图像数据。这发生在`DecodeAndProcessImage()`函数中：

```py
  TfLiteStatus decode_status = DecodeAndProcessImage(
      error_reporter, image_width, image_height, image_data);
```

该函数使用一个名为JPEGDecoder的库来解码JPEG数据，并直接将其写入模型的输入张量。在此过程中，它裁剪图像，丢弃一些160×120的数据，使剩下的只有96×96像素，大致位于图像中心。它还将图像的16位颜色表示减少到8位灰度。

在图像被捕获并存储在输入张量中后，我们准备运行推理。接下来，我们展示模型的输出是如何显示的。

### 在Arduino上响应检测

Arduino Nano 33 BLE Sense内置了RGB LED，这是一个包含独立红色、绿色和蓝色LED的单一组件，您可以分别控制它们。检测响应器的实现在每次推理运行时闪烁蓝色LED。当检测到人时，点亮绿色LED；当未检测到人时，点亮红色LED。

实现在[*arduino/detection_responder.cc*](https://oreil.ly/-WsSN)中。让我们快速浏览一下。

`RespondToDetection()`函数接受两个分数，一个用于“人”类别，另一个用于“非人”。第一次调用时，它设置蓝色、绿色和黄色LED为输出：

```py
void RespondToDetection(tflite::ErrorReporter* error_reporter,
                        uint8_t person_score, uint8_t no_person_score) {
  static bool is_initialized = false;
  if (!is_initialized) {
    pinMode(led_green, OUTPUT);
    pinMode(led_blue, OUTPUT);
    is_initialized = true;
  }
```

接下来，为了指示推理刚刚完成，我们关闭所有LED，然后非常简要地闪烁蓝色LED：

```py
  // Note: The RGB LEDs on the Arduino Nano 33 BLE
  // Sense are on when the pin is LOW, off when HIGH.

  // Switch the person/not person LEDs off
  digitalWrite(led_green, HIGH);
  digitalWrite(led_red, HIGH);

  // Flash the blue LED after every inference.
  digitalWrite(led_blue, LOW);
  delay(100);
  digitalWrite(led_blue, HIGH);
```

您会注意到，与Arduino内置LED不同，这些LED使用`LOW`打开，使用`HIGH`关闭。这只是LED连接到板上的方式的一个因素。

接下来，我们根据哪个类别的分数更高来打开和关闭适当的LED：

```py
  // Switch on the green LED when a person is detected,
  // the red when no person is detected
  if (person_score > no_person_score) {
    digitalWrite(led_green, LOW);
    digitalWrite(led_red, HIGH);
  } else {
    digitalWrite(led_green, HIGH);
    digitalWrite(led_red, LOW);
  }
```

最后，我们使用`error_reporter`实例将分数输出到串行端口：

```py
  error_reporter->Report("Person score: %d No person score: %d", person_score,
                         no_person_score);
}
```

就是这样！函数的核心是一个基本的`if`语句，您可以轻松使用类似的逻辑来控制其他类型的输出。将如此复杂的视觉输入转换为一个布尔输出“人”或“非人”是非常令人兴奋的事情。

### 运行示例

运行此示例比我们其他Arduino示例更复杂，因为我们需要将Arducam连接到Arduino板。我们还需要安装和配置与Arducam接口并解码其JPEG输出的库。但不用担心，这仍然非常简单！

要部署此示例，我们需要以下内容：

+   一个Arduino Nano 33 BLE Sense板

+   一个Arducam Mini 2MP Plus

+   跳线（和可选的面包板）

+   一根Micro-USB电缆

+   Arduino IDE

我们的第一个任务是使用跳线连接Arducam到Arduino。这不是一本电子书，所以我们不会详细介绍使用电缆的细节。相反，[表9-1](#arducam_pins)显示了引脚应该如何连接。每个设备上都标有引脚标签。

表9-1。Arducam Mini 2MP Plus到Arduino Nano 33 BLE Sense的连接

| Arducam引脚 | Arduino引脚 |
| --- | --- |
| CS | D7（未标记，紧挨D6右侧） |
| MOSI | D11 |
| MISO | D12 |
| SCK | D13 |
| GND | GND（任何一个标记为GND的引脚都可以） |
| VCC | 3.3 V |
| SDA | A4 |
| SCL | A5 |

设置硬件后，您可以继续安装软件。

###### 提示

建立过程可能会有所变化，所以请查看[*README.md*](https://oreil.ly/CR5Pb)获取最新说明。

本书中的项目作为TensorFlow Lite Arduino库中的示例代码可用。如果您尚未安装该库，请打开Arduino IDE并从“工具”菜单中选择“管理库”。在弹出的窗口中，搜索并安装名为*Arduino_TensorFlowLite*的库。您应该能够使用最新版本，但如果遇到问题，本书测试过的版本是1.14-ALPHA。

###### 注意

您还可以从*.zip*文件安装库，您可以从TensorFlow Lite团队[下载](https://oreil.ly/blgB8)或使用TensorFlow Lite for Microcontrollers Makefile自动生成。如果您更喜欢后者，请参阅[附录A](app01.xhtml#appendix_arduino_library_zip)。

安装完库后，`person_detection`示例将显示在“文件”菜单下的“示例→Arduino_TensorFlowLite”中，如[图9-2](#arduino_examples_person_detection)所示。

![“示例”菜单的屏幕截图](Images/timl_0604.png)

###### 图9-2。示例菜单

点击“person_detection”加载示例。它将显示为一个新窗口，每个源文件都有一个选项卡。第一个选项卡中的文件*person_detection*相当于我们之前介绍的*main_functions.cc*。

###### 注意

[“运行示例”](ch06.xhtml#hello_world_running_the_example)已经解释了Arduino示例的结构，所以我们这里不再重复覆盖。

除了TensorFlow库，我们还需要安装另外两个库：

+   Arducam库，以便我们的代码可以与硬件进行交互

+   JPEGDecoder库，以便我们可以解码JPEG编码的图像

Arducam Arduino库可从[GitHub](https://oreil.ly/93OKK)获取。要安装它，请下载或克隆存储库。接下来，将其*ArduCAM*子目录复制到*Arduino/libraries*目录中。要找到您机器上的*libraries*目录，请在Arduino IDE的首选项窗口中检查Sketchbook位置。

下载库后，您需要编辑其中一个文件，以确保为Arducam Mini 2MP Plus进行配置。为此，请打开*Arduino/libraries/ArduCAM/memorysaver.h*。

您会看到一堆`#define`语句。确保它们都被注释掉，除了`#define OV2640_MINI_2MP_PLUS`，如此处所示：

```py
//Step 1: select the hardware platform, only one at a time
//#define OV2640_MINI_2MP
//#define OV3640_MINI_3MP
//#define OV5642_MINI_5MP
//#define OV5642_MINI_5MP_BIT_ROTATION_FIXED
#define OV2640_MINI_2MP_PLUS
//#define OV5642_MINI_5MP_PLUS
//#define OV5640_MINI_5MP_PLUS
```

保存文件后，您已经完成了Arducam库的配置。

###### 提示

示例是使用Arducam库的提交#e216049开发的。如果您在使用库时遇到问题，可以尝试下载这个特定的提交，以确保您使用的是完全相同的代码。

下一步是安装JPEGDecoder库。您可以在Arduino IDE中完成这个操作。在工具菜单中，选择管理库选项并搜索JPEGDecoder。您应该安装库的1.8.0版本。

安装完库之后，您需要配置它以禁用一些与Arduino Nano 33 BLE Sense不兼容的可选组件。打开*Arduino/libraries/JPEGDecoder/src/User_Config.h*，确保`#define LOAD_SD_LIBRARY`和`#define LOAD_SDFAT_LIBRARY`都被注释掉，如文件中的摘录所示：

```py
// Comment out the next #defines if you are not using an SD Card to store
// the JPEGs
// Commenting out the line is NOT essential but will save some FLASH space if
// SD Card access is not needed. Note: use of SdFat is currently untested!

//#define LOAD_SD_LIBRARY // Default SD Card library
//#define LOAD_SDFAT_LIBRARY // Use SdFat library instead, so SD Card SPI can
                             // be bit bashed
```

保存文件后，安装库就完成了。现在您已经准备好运行人员检测应用程序了！

首先，通过USB将Arduino设备插入。确保在工具菜单中从板下拉列表中选择正确的设备类型，如[图9-3](#arduino_board_dropdown_9)所示。

![“板”下拉列表的截图](Images/timl_0605.png)

###### 图9-3. 板下拉列表

如果您的设备名称不在列表中显示，您需要安装其支持包。要做到这一点，请点击Boards Manager。在弹出的窗口中搜索您的设备并安装相应支持包的最新版本。

在工具菜单中，还要确保设备的端口在端口下拉列表中被选中，如[图9-4](#arduino_port_dropdown_9)所示。

![“端口”下拉列表的截图](Images/timl_0606.png)

###### 图9-4. 端口下拉列表

最后，在Arduino窗口中，点击上传按钮（在[图9-5](#arduino_upload_button_9)中用白色标出）来编译并上传代码到您的Arduino设备。

![上传按钮的截图](Images/timl_0607.png)

###### 图9-5. 上传按钮

一旦上传成功完成，程序将运行。

要测试它，首先将设备的摄像头对准明显不是人的东西，或者只是遮住镜头。下次蓝色LED闪烁时，设备将从摄像头捕获一帧并开始运行推理。由于我们用于人员检测的视觉模型相对较大，这将需要很长时间的推理——在撰写本文时大约需要19秒，尽管自那时起TensorFlow Lite可能已经变得更快。

当推断完成时，结果将被翻译为另一个LED被点亮。您将相机对准了一个不是人的东西，所以红色LED应该点亮。

现在，尝试将设备的相机对准自己！下次蓝色LED闪烁时，设备将捕获另一幅图像并开始运行推断。大约19秒后，绿色LED应该亮起。

请记住，在每次推断之前，图像数据都会被捕获为快照，每当蓝色LED闪烁时。在那一刻相机对准的东西将被馈送到模型中。在下一次捕获图像时，相机对准的位置并不重要，当蓝色LED再次闪烁时，图像将被捕获。

如果您得到看似不正确的结果，请确保您处于光线良好的环境中。您还应确保相机的方向正确，引脚朝下，以便捕获的图像是正确的方式——该模型没有经过训练以识别颠倒的人。此外，值得记住这是一个微小的模型，它以小尺寸换取准确性。它工作得非常好，但并非100%准确。

您还可以通过Arduino串行监视器查看推断的结果。要做到这一点，请从“工具”菜单中打开串行监视器。您将看到一个详细的日志，显示应用程序运行时发生的情况。还有一个有趣的功能是勾选“显示时间戳”框，这样您就可以看到每个过程需要多长时间：

```py
14:17:50.714 -> Starting capture
14:17:50.714 -> Image captured
14:17:50.784 -> Reading 3080 bytes from ArduCAM
14:17:50.887 -> Finished reading
14:17:50.887 -> Decoding JPEG and converting to greyscale
14:17:51.074 -> Image decoded and processed
14:18:09.710 -> Person score: 246 No person score: 66
```

从这个日志中，我们可以看到从相机模块捕获和读取图像数据大约需要170毫秒，解码JPEG并将其转换为灰度需要180毫秒，运行推断需要18.6秒。

### 进行自己的更改

现在您已部署了基本应用程序，请尝试玩耍并对代码进行一些更改。只需在Arduino IDE中编辑文件并保存，然后重复之前的说明以将修改后的代码部署到设备上。

以下是您可以尝试的几件事：

+   修改检测响应器，使其忽略模糊的输入，即“人”和“无人”得分之间没有太大差异的情况。

+   使用人员检测的结果来控制其他组件，如额外的LED或伺服。

+   构建一个智能安全摄像头，通过存储或传输图像来实现，但仅限于包含人物的图像。

## SparkFun Edge

SparkFun Edge板经过优化，以实现低功耗。当与同样高效的相机模块配对时，它是构建视觉应用程序的理想平台，这些应用程序将在电池供电时运行。通过板上的排线适配器轻松插入相机模块。

### 要购买哪种相机模块

此示例使用SparkFun的[Himax HM01B0分支相机模块](https://oreil.ly/H24xS)。它基于一个320×320像素的图像传感器，当以每秒30帧的速度捕获时，消耗极少的功率：不到2 mW。

### 在SparkFun Edge上捕获图像

要开始使用Himax HM01B0相机模块捕获图像，我们首先必须初始化相机。完成此操作后，我们可以在需要新图像时从相机读取一帧。一帧是一个表示相机当前所看到的内容的字节数组。

使用相机将涉及大量使用Ambiq Apollo3 SDK和HM01B0驱动程序，后者作为构建过程的一部分下载，位于[*sparkfun_edge/himax_driver*](https://oreil.ly/OhBj0)中。

图像提供程序实现在[*sparkfun_edge/image_provider.cc*](https://oreil.ly/ZdU9N)中。我们不会解释其每个细节，因为代码是针对SparkFun板和Himax相机模块的。相反，让我们以高层次的方式来看看发生了什么。

`GetImage()` 函数是图像提供程序与世界的接口。它在我们的应用程序的主循环中被调用以获取一帧图像数据。第一次调用时，我们需要初始化摄像头。这通过调用 `InitCamera()` 函数来实现，如下所示：

```py
// Capture single frame.  Frame pointer passed in to reduce memory usage.  This
// allows the input tensor to be used instead of requiring an extra copy.
TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int frame_width,
                      int frame_height, int channels, uint8_t* frame) {
  if (!g_is_camera_initialized) {
    TfLiteStatus init_status = InitCamera(error_reporter);
    if (init_status != kTfLiteOk) {
      am_hal_gpio_output_set(AM_BSP_GPIO_LED_RED);
      return init_status;
    }
```

如果 `InitCamera()` 返回除了 `kTfLiteOk` 状态之外的任何内容，我们会打开板上的红色 LED（使用 `am_hal_gpio_output_set(AM_BSP_GPIO_LED_RED)`）来指示问题。这对于调试很有帮助。

`InitCamera()` 函数在 *image_provider.cc* 中进一步定义。我们不会在这里详细介绍它，因为它非常特定于设备，如果您想在自己的代码中使用它，只需复制粘贴即可。

它调用一堆 Apollo3 SDK 函数来配置微控制器的输入和输出，以便它可以与摄像头模块通信。它还启用了*中断*，这是摄像头用来发送新图像数据的机制。当这一切设置完成后，它使用摄像头驱动程序打开摄像头，并配置它开始持续捕获图像。

摄像头模块具有自动曝光功能，它会在捕获帧时自动校准曝光设置。为了让它有机会在我们尝试执行推理之前校准，`GetImage()` 函数的下一部分使用摄像头驱动程序的 `hm01b0_blocking_read_oneframe_scaled()` 函数捕获几帧图像。我们不对捕获的数据做任何处理；我们只是为了让摄像头模块的自动曝光功能有一些材料可以使用：

```py
    // Drop a few frames until auto exposure is calibrated.
    for (int i = 0; i < kFramesToInitialize; ++i) {
      hm01b0_blocking_read_oneframe_scaled(frame, frame_width, frame_height,
                                           channels);
    }
    g_is_camera_initialized = true;
  }
```

设置完成后，`GetImage()` 函数的其余部分非常简单。我们只需调用 `hm01b0_blocking_read_oneframe_scaled()` 来捕获一幅图像：

```py
hm01b0_blocking_read_oneframe_scaled(frame, frame_width, frame_height,
                                     channels);
```

当应用程序的主循环中调用 `GetImage()` 时，`frame` 变量是指向我们输入张量的指针，因此数据直接由摄像头驱动程序写入到为输入张量分配的内存区域。我们还指定了我们想要的宽度、高度和通道数。

通过这个实现，我们能够从我们的摄像头模块中捕获图像数据。接下来，让我们看看如何响应模型的输出。

### 在 SparkFun Edge 上响应检测

检测响应器的实现与我们的唤醒词示例的命令响应器非常相似。每次运行推理时，它会切换设备的蓝色 LED。当检测到一个人时，它会点亮绿色 LED，当没有检测到一个人时，它会点亮黄色 LED。

实现在 [*sparkfun_edge/detection_responder.cc*](https://oreil.ly/OeN1M) 中。让我们快速浏览一下。

`RespondToDetection()` 函数接受两个分数，一个用于“人”类别，另一个用于“非人”。第一次调用时，它会为蓝色、绿色和黄色 LED 设置输出：

```py
void RespondToDetection(tflite::ErrorReporter* error_reporter,
                        uint8_t person_score, uint8_t no_person_score) {
  static bool is_initialized = false;
  if (!is_initialized) {
    // Setup LED's as outputs.  Leave red LED alone since that's an error
    // indicator for sparkfun_edge in image_provider.
    am_hal_gpio_pinconfig(AM_BSP_GPIO_LED_BLUE, g_AM_HAL_GPIO_OUTPUT_12);
    am_hal_gpio_pinconfig(AM_BSP_GPIO_LED_GREEN, g_AM_HAL_GPIO_OUTPUT_12);
    am_hal_gpio_pinconfig(AM_BSP_GPIO_LED_YELLOW, g_AM_HAL_GPIO_OUTPUT_12);
    is_initialized = true;
  }
```

因为该函数每次推理调用一次，所以下面的代码片段会导致它在每次执行推理时切换蓝色 LED 的开关：

```py
// Toggle the blue LED every time an inference is performed.
static int count = 0;
if (++count & 1) {
    am_hal_gpio_output_set(AM_BSP_GPIO_LED_BLUE);
} else {
    am_hal_gpio_output_clear(AM_BSP_GPIO_LED_BLUE);
}
```

最后，如果检测到一个人，它会点亮绿色 LED，如果没有检测到一个人，它会点亮蓝色 LED。它还使用 `ErrorReporter` 实例记录分数：

```py
am_hal_gpio_output_clear(AM_BSP_GPIO_LED_YELLOW);
am_hal_gpio_output_clear(AM_BSP_GPIO_LED_GREEN);
if (person_score > no_person_score) {
    am_hal_gpio_output_set(AM_BSP_GPIO_LED_GREEN);
} else {
    am_hal_gpio_output_set(AM_BSP_GPIO_LED_YELLOW);
}

error_reporter->Report("person score:%d no person score %d", person_score,
                        no_person_score);
```

就是这样！函数的核心是一个基本的 `if` 语句，你可以很容易地使用类似的逻辑来控制其他类型的输出。将如此复杂的视觉输入转换为一个布尔输出“人”或“非人”是非常令人兴奋的事情。

### 运行示例

现在我们已经看到了 SparkFun Edge 实现的工作原理，让我们开始运行它。

###### 提示

由于本书编写时可能已更改构建过程，因此请查看 [*README.md*](https://oreil.ly/kaSXN) 获取最新说明。

要构建和部署我们的代码，我们需要以下内容：

+   带有 [Himax HM01B0 breakout](https://oreil.ly/jNtyv) 的 SparkFun Edge 开发板

+   一个USB编程器（我们推荐SparkFun串行基础分支，可在[micro-B USB](https://oreil.ly/wXo-f)和[USB-C](https://oreil.ly/-YvfN)变种中获得）

+   一根匹配的USB电缆

+   Python 3和一些依赖项

###### 注意

如果您不确定是否安装了正确版本的Python，请参考[“运行示例”](ch06.xhtml#running_hello_world_sparkfun_edge)中的说明进行检查。

在终端中，克隆TensorFlow存储库并切换到其目录：

```py
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
```

接下来，我们将构建二进制文件并运行一些命令，使其准备好下载到设备上。为了避免一些打字，您可以从[*README.md*](https://oreil.ly/kaSXN)中复制并粘贴这些命令。

#### 构建二进制文件

以下命令下载所有必需的依赖项，然后为SparkFun Edge编译一个二进制文件：

```py
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=sparkfun_edge person_detection_bin
```

二进制文件被创建为*.bin*文件，位于以下位置：

```py
tensorflow/lite/micro/tools/make/gen/
  sparkfun_edge_cortex-m4/bin/person_detection.bin
```

要检查文件是否存在，可以使用以下命令：

```py
test -f tensorflow/lite/micro/tools/make/gen \
  /sparkfun_edge_cortex-m4/bin/person_detection.bin \
  &&  echo "Binary was successfully created" || echo "Binary is missing"
```

当您运行该命令时，您应该看到`Binary was successfully created`打印到控制台。

如果看到`Binary is missing`，则构建过程中出现问题。如果是这样，很可能在`make`命令的输出中有一些线索指出出了什么问题。

#### 对二进制文件进行签名

必须使用加密密钥对二进制文件进行签名，才能部署到设备上。现在让我们运行一些命令，对二进制文件进行签名，以便可以刷写到SparkFun Edge上。这里使用的脚本来自Ambiq SDK，在运行Makefile时下载。

输入以下命令设置一些虚拟的加密密钥，供开发使用：

```py
cp tensorflow/lite/micro/tools/make/downloads/AmbiqSuite-Rel2.0.0 \
  /tools/apollo3_scripts/keys_info0.py \
tensorflow/lite/micro/tools/make/downloads/AmbiqSuite-Rel2.0.0 \
  /tools/apollo3_scripts/keys_info.py
```

接下来，运行以下命令创建一个已签名的二进制文件。如果需要，将`python3`替换为`python`：

```py
python3 tensorflow/lite/micro/tools/make/downloads/ \
  AmbiqSuite-Rel2.0.0/tools/apollo3_scripts/create_cust_image_blob.py \
  --bin tensorflow/lite/micro/tools/make/gen/ \
  sparkfun_edge_cortex-m4/bin/person_detection.bin \
  --load-address 0xC000 \
  --magic-num 0xCB \
  -o main_nonsecure_ota \
  --version 0x0
```

这将创建文件*main_nonsecure_ota.bin*。现在运行此命令创建文件的最终版本，您可以使用该文件刷写设备，使用下一步中将使用的脚本：

```py
python3 tensorflow/lite/micro/tools/make/downloads/ \
  AmbiqSuite-Rel2.0.0/tools/apollo3_scripts/create_cust_wireupdate_blob.py \
  --load-address 0x20000 \
  --bin main_nonsecure_ota.bin \
  -i 6 \
  -o main_nonsecure_wire \
  --options 0x1
```

现在您应该在运行命令的目录中有一个名为*main_nonsecure_wire.bin*的文件。这是您将要刷写到设备的文件。

#### 刷写二进制文件

SparkFun Edge将当前运行的程序存储在其1兆字节的闪存中。如果您希望板运行新程序，您需要将其发送到板上，该程序将存储在闪存中，覆盖先前保存的任何程序。

正如我们在本书中早些时候提到的，这个过程被称为*刷写*。

#### 将编程器连接到板上

要下载新程序到板上，您将使用SparkFun USB-C串行基础串行编程器。该设备允许您的计算机通过USB与微控制器通信。

将此设备连接到您的板上，执行以下步骤：

1.  在SparkFun Edge的一侧，找到六针排针。

1.  将SparkFun USB-C串行基础插入这些引脚，确保每个设备上标记为BLK和GRN的引脚正确对齐，如[图9-6](#sparkfun_edge_serial_basic_3)所示。

![显示SparkFun Edge和USB-C串行基础如何连接的照片](Images/timl_0613.png)

###### 图9-6\. 连接SparkFun Edge和USB-C串行基础（由SparkFun提供）

#### 将编程器连接到计算机

通过USB将板连接到计算机。要对板进行编程，您需要找出计算机给设备的名称。最好的方法是在连接设备之前和之后列出所有计算机的设备，然后查看哪个设备是新的。

###### 警告

一些人报告了他们操作系统的默认驱动程序与编程器存在问题，因此我们强烈建议在继续之前安装[驱动程序](https://oreil.ly/yI-NR)。

在通过USB连接设备之前，运行以下命令：

```py
# macOS:
ls /dev/cu*

# Linux:
ls /dev/tty*
```

这应该输出一个附加设备列表，看起来像以下内容：

```py
/dev/cu.Bluetooth-Incoming-Port
/dev/cu.MALS
/dev/cu.SOC
```

现在，将编程器连接到计算机的USB端口，并再次运行以下命令：

```py
# macOS:
ls /dev/cu*

# Linux:
ls /dev/tty*
```

您应该在输出中看到一个额外的项目，如下例所示。您的新项目可能有不同的名称。这个新项目是设备的名称：

```py
/dev/cu.Bluetooth-Incoming-Port
/dev/cu.MALS
/dev/cu.SOC
/dev/cu.wchusbserial-1450
```

这个名称将用于引用设备。但是，它可能会根据编程器连接的USB端口而变化，因此如果您将板子从计算机断开然后重新连接，可能需要再次查找其名称。

###### 提示

一些用户报告列表中出现了两个设备。如果看到两个设备，则要使用的正确设备以“wch”开头；例如，`/dev/wchusbserial-14410.`

在确定设备名称后，将其放入一个shell变量以备后用：

```py
export DEVICENAME=<*your device name here*>

```

这是在后续过程中运行需要设备名称的命令时可以使用的变量。

#### 运行脚本以刷写您的板子

要刷写板子，您需要将其置于特殊的“引导加载程序”状态，以准备接收新的二进制文件。然后，您将运行一个脚本将二进制文件发送到板子。

首先创建一个环境变量来指定波特率，即数据发送到设备的速度：

```py
export BAUD_RATE=921600
```

现在将以下命令粘贴到终端中，但*不要立即按Enter*！命令中的`${DEVICENAME}`和`${BAUD_RATE}`将被替换为您在前面部分设置的值。如有必要，请记得将`python3`替换为`python`。

```py
python3 tensorflow/lite/micro/tools/make/downloads/ \
  AmbiqSuite-Rel2.0.0/tools/apollo3_scripts/uart_wired_update.py -b \
  ${BAUD_RATE} ${DEVICENAME} -r 1 -f main_nonsecure_wire.bin -i 6
```

接下来，您将重置板子到引导加载程序状态并刷写板子。在板子上，找到标有`RST`和`14`的按钮，如[图9-7](#sparkfun_edge_buttons_3)所示。

执行以下步骤：

1.  确保您的板子连接到编程器，并且整个设备通过USB连接到计算机。

1.  在板子上，按住标有`14`的按钮。*继续按住它。*

1.  在仍按住标有`14`的按钮的情况下，按下标有`RST`的按钮重置板子。

1.  在计算机上按Enter运行脚本。*继续按住按钮`14`。*

![显示SparkFun Edge按钮的照片](Images/timl_0614.png)

###### 图9-7\. SparkFun Edge的按钮

您现在应该在屏幕上看到类似以下内容：

```py
Connecting with Corvette over serial port /dev/cu.usbserial-1440...
Sending Hello.
Received response for Hello
Received Status
length =  0x58
version =  0x3
Max Storage =  0x4ffa0
Status =  0x2
State =  0x7
AMInfo =
0x1
0xff2da3ff
0x55fff
0x1
0x49f40003
0xffffffff
[...lots more 0xffffffff...]
Sending OTA Descriptor =  0xfe000
Sending Update Command.
number of updates needed =  1
Sending block of size  0x158b0  from  0x0  to  0x158b0
Sending Data Packet of length  8180
Sending Data Packet of length  8180
[...lots more Sending Data Packet of length  8180...]
```

继续按住按钮`14`，直到看到`Sending Data Packet of length 8180`。在看到此信息后可以释放按钮（但如果继续按住也没关系）。

程序将继续在终端上打印行。最终，您会看到类似以下内容：

```py
[...lots more Sending Data Packet of length  8180...]
Sending Data Packet of length  8180
Sending Data Packet of length  6440
Sending Reset Command.
Done.
```

这表示刷写成功。

###### 提示

如果程序输出以错误结束，请检查是否打印了`Sending Reset Command.`。如果是，则尽管出现错误，刷写可能已成功。否则，刷写可能失败。尝试再次运行这些步骤（您可以跳过设置环境变量）。

### 测试程序

首先按下`RST`按钮，确保程序正在运行。

当程序运行时，蓝色LED将交替闪烁，每次推理一次。由于我们用于人员检测的视觉模型相对较大，运行推理需要很长时间——总共约6秒。

首先将设备的摄像头对准绝对不是人的东西，或者只是遮住镜头。下一次蓝色LED切换时，设备将从摄像头捕获一帧并开始运行推理。大约6秒后，推理结果将被转换为另一个LED点亮。鉴于您将摄像头对准的不是人，橙色LED应该点亮。

现在，尝试将设备的摄像头对准自己。下一次蓝色LED切换时，设备将捕获另一帧并开始运行推理。这次，绿色LED应该点亮。

请记住，在每次推理之前，图像数据都会被捕获为快照，每当蓝色LED切换时。在那一刻摄像头对准的东西将被输入模型。在下一次捕获帧时，摄像头对准的位置并不重要，蓝色LED将再次切换。

如果您得到看似不正确的结果，请确保您处于光线良好的环境中。还要记住，这是一个小模型，它以精度换取了小尺寸。它工作得非常好，但并非始终100%准确。

### 查看调试数据

该程序将检测结果记录到串行端口。要查看它们，我们可以使用波特率为115200监视板的串行端口输出。在macOS和Linux上，以下命令应该有效：

```py
screen ${DEVICENAME} 115200
```

您应该最初看到类似以下内容的输出：

```py
Apollo3 Burst Mode is Available

                               Apollo3 operating in Burst Mode (96MHz)
```

当板捕获帧并运行推断时，您应该看到它打印调试信息：

```py
Person score: 130 No person score: 204
Person score: 220 No person score: 87
```

要停止使用`screen`查看调试输出，请按Ctrl-A，紧接着按K键，然后按Y键。

### 进行您自己的更改

现在您已经部署了基本应用程序，请尝试玩耍并进行一些更改。您可以在*tensorflow/lite/micro/examples/person_detection*文件夹中找到应用程序的代码。只需编辑并保存，然后重复前面的说明以将修改后的代码部署到设备上。

以下是您可以尝试的一些事项：

+   修改检测响应器，使其忽略模糊的输入，即“人”和“无人”得分之间没有太大差异的情况。

+   利用人员检测结果来控制其他组件，如额外的LED或伺服。

+   构建一个智能安全摄像头，通过存储或传输图像来检测只包含人的图像。

# 总结

我们在本章中使用的视觉模型是一件了不起的事情。它接受原始且混乱的输入，无需预处理，并为我们提供一个非常简单的输出：是，有人在场，还是，没有人在场。这就是机器学习的魔力：它可以从噪音中过滤信息，留下我们关心的信号。作为开发者，我们可以轻松使用这些信号为用户构建令人惊叹的体验。

在构建机器学习应用程序时，很常见使用像这样的预训练模型，这些模型已经包含执行任务所需的知识。模型大致相当于代码库，封装了特定功能，并且可以在项目之间轻松共享。您经常会发现自己在探索和评估模型，寻找适合您任务的合适模型。

在[第10章](ch10.xhtml#chapter_person_detection_training)中，我们将探讨人员检测模型的工作原理。您还将学习如何训练自己的视觉模型来识别不同类型的对象。

在[2018年YouGov民意调查](https://oreil.ly/KvzGk)中，70%的受访者表示，如果失去视力，他们会最怀念视觉。
