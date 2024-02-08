# 第六章：TinyML 的“Hello World”：部署到微控制器

现在是时候动手了。在本章的过程中，我们将代码部署到三种不同的设备上：

+   [Arduino Nano 33 BLE Sense](https://oreil.ly/6qlMD)

+   [SparkFun Edge](https://oreil.ly/-hoL-)

+   [ST Microelectronics STM32F746G Discovery kit](https://oreil.ly/cvm4J)

我们将逐个讨论每个设备的构建和部署过程。

###### 注意

TensorFlow Lite 定期添加对新设备的支持，因此如果您想使用的设备未在此处列出，值得查看示例的[*README.md*](https://oreil.ly/ez0ef)。

如果在按照这些步骤时遇到问题，您也可以在那里查看更新的部署说明。

每个设备都有自己独特的输出能力，从一组 LED 到完整的 LCD 显示器，因此示例包含每个设备的`HandleOutput()`的自定义实现。我们还将逐个讨论这些，并谈谈它们的逻辑如何工作。即使您没有所有这些设备，阅读这段代码也应该很有趣，因此我们强烈建议您查看一下。

# 什么是微控制器？

根据您的过去经验，您可能不熟悉微控制器如何与其他电子组件交互。因为我们即将开始玩硬件，所以在继续之前介绍一些概念是值得的。

在像 Arduino、SparkFun Edge 或 STM32F746G Discovery kit 这样的微控制器板上，实际的微控制器只是连接到电路板的许多电子组件之一。图 6-1 显示了 SparkFun Edge 上的微控制器。

![突出显示其微控制器的 SparkFun Edge 板的图像](img/timl_0601.png)

###### 图 6-1。SparkFun Edge 板上突出显示其微控制器

微控制器使用*引脚*连接到其所在的电路板。典型的微控制器有数十个引脚，它们有各种用途。一些引脚为微控制器提供电源；其他连接到各种重要组件。一些引脚专门用于由运行在微控制器上的程序输入和输出数字信号。这些被称为*GPIO*引脚，代表通用输入/输出。它们可以作为输入，确定是否向其施加电压，或作为输出，提供可以为其他组件供电或通信的电流。

GPIO 引脚是数字的。这意味着在输出模式下，它们就像开关，可以完全打开或完全关闭。在输入模式下，它们可以检测由其他组件施加在它们上的电压是高于还是低于某个阈值。

除了 GPIO，一些微控制器还具有模拟输入引脚，可以测量施加在它们上的电压的确切水平。

通过调用特殊函数，运行在微控制器上的程序可以控制特定引脚是输入模式还是输出模式。其他函数用于打开或关闭输出引脚，或读取输入引脚的当前状态。

现在您对微控制器有了更多了解，让我们更仔细地看看我们的第一个设备：Arduino。

# Arduino

有各种各样的[Arduino](https://www.arduino.cc/)板，具有不同的功能。并非所有板都能运行 TensorFlow Lite for Microcontrollers。我们推荐本书使用的板是[Arduino Nano 33 BLE Sense](https://oreil.ly/9g1bJ)。除了与 TensorFlow Lite 兼容外，它还包括麦克风和加速度计（我们将在后面的章节中使用）。我们建议购买带有引脚排针的板，这样可以更容易地连接其他组件而无需焊接。

大多数 Arduino 板都带有内置 LED，这就是我们将用来可视化输出正弦值的内容。图 6-2 显示了一个 Arduino Nano 33 BLE Sense 板，其中突出显示了 LED。

![突出显示的 Arduino Nano 33 BLE Sense 板上的 LED 的图像](img/timl_0602.png)

###### 图 6-2。Arduino Nano 33 BLE Sense 板上突出显示的 LED

## 在 Arduino 上处理输出

因为我们只有一个 LED 可供使用，所以我们需要进行创造性思考。一种选择是根据最近预测的正弦值来改变 LED 的亮度。鉴于该值范围为-1 到 1，我们可以用完全关闭的 LED 表示 0，用完全亮起的 LED 表示-1 和 1，用部分调暗的 LED 表示任何中间值。当程序在循环中运行推断时，LED 将重复地变暗和变亮。

我们可以使用`kInferencesPerCycle`常量在完整正弦波周期内执行的推断数量。由于一个推断需要一定的时间，调整*constants.cc*中定义的`kInferencesPerCycle`将调整 LED 变暗的速度。

在[*hello_world/arduino/constants.cc*](https://oreil.ly/YNsvq)中有一个特定于 Arduino 的版本的此文件。该文件与*hello_world/constants.cc*具有相同的名称，因此在为 Arduino 构建应用程序时将使用它来代替原始实现。

为了调暗我们的内置 LED，我们可以使用一种称为*脉宽调制*（PWM）的技术。如果我们非常快速地打开和关闭一个输出引脚，那么引脚的输出电压将成为处于关闭和打开状态之间所花时间比率的因素。如果引脚在每种状态中花费的时间占 50%，则其输出电压将是其最大值的 50%。如果它在打开状态花费 75%的时间，关闭状态花费 25%的时间，则其电压将是其最大值的 75%。

PWM 仅在某些 Arduino 设备的某些引脚上可用，但使用起来非常简单：我们只需调用一个设置所需输出电平的函数。

用于 Arduino 输出处理的代码位于[*hello_world/arduino/output_handler.cc*](https://oreil.ly/OpLMB)中，该代码用于替代原始文件*hello_world/output_handler.cc*。

让我们来看一下源代码：

```py
#include "tensorflow/lite/micro/examples/hello_world/output_handler.h"
#include "Arduino.h"
#include "tensorflow/lite/micro/examples/hello_world/constants.h"
```

首先，我们包含一些头文件。我们的*output_handler.h*指定了此文件的接口。*Arduino.h*提供了 Arduino 平台的接口；我们使用它来控制板。因为我们需要访问`kInferencesPerCycle`，所以我们还包括*constants.h*。

接下来，我们定义函数并指示它第一次运行时要执行的操作：

```py
// Adjusts brightness of an LED to represent the current y value
void HandleOutput(tflite::ErrorReporter* error_reporter, float x_value,
                  float y_value) {
// Track whether the function has run at least once
static bool is_initialized = false;

// Do this only once
if (!is_initialized) {
  // Set the LED pin to output
  pinMode(LED_BUILTIN, OUTPUT);
  is_initialized = true;
}
```

在 C++中，函数内声明为`static`的变量将在函数的多次运行中保持其值。在这里，我们使用`is_initialized`变量来跟踪以下`if (!is_initialized)`块中的代码是否曾经运行过。

初始化块调用 Arduino 的[`pinMode()`](https://oreil.ly/6Kxep)函数，该函数指示微控制器给定的引脚应该处于输入模式还是输出模式。在使用引脚之前，这是必要的。该函数使用 Arduino 平台定义的两个常量调用：`LED_BUILTIN`和`OUTPUT`。`LED_BUILTIN`表示连接到板上内置 LED 的引脚，`OUTPUT`表示输出模式。

将内置 LED 的引脚配置为输出模式后，将`is_initialized`设置为`true`，以便此代码块不会再次运行。

接下来，我们计算 LED 的期望亮度：

```py
// Calculate the brightness of the LED such that y=-1 is fully off
// and y=1 is fully on. The LED's brightness can range from 0-255.
int brightness = (int)(127.5f * (y_value + 1));
```

Arduino 允许我们将 PWM 输出的电平设置为 0 到 255 之间的数字，其中 0 表示完全关闭，255 表示完全打开。我们的`y_value`是-1 到 1 之间的数字。前面的代码将`y_value`映射到 0 到 255 的范围，因此当`y = -1`时，LED 完全关闭，当`y = 0`时，LED 亮度为一半，当`y = 1`时，LED 完全亮起。

下一步是实际设置 LED 的亮度：

```py
// Set the brightness of the LED. If the specified pin does not support PWM,
// this will result in the LED being on when y > 127, off otherwise.
analogWrite(LED_BUILTIN, brightness);
```

Arduino 平台的[`analogWrite()`](https://oreil.ly/nNseR)函数接受一个介于 0 和 255 之间的值的引脚号（我们提供`LED_BUILTIN`）和我们在前一行中计算的`brightness`。当调用此函数时，LED 将以该亮度点亮。

###### 注意

不幸的是，在某些型号的 Arduino 板上，内置 LED 连接的引脚不支持 PWM。这意味着我们对`analogWrite()`的调用不会改变其亮度。相反，如果传递给`analogWrite()`的值大于 127，则 LED 将打开，如果小于等于 126，则 LED 将关闭。这意味着 LED 将闪烁而不是渐变。虽然不够酷，但仍然展示了我们的正弦波预测。

最后，我们使用`ErrorReporter`实例记录亮度值：

```py
// Log the current brightness value for display in the Arduino plotter
error_reporter->Report("%d\n", brightness);
```

在 Arduino 平台上，`ErrorReporter`被设置为通过串行端口记录数据。串行是微控制器与主机计算机通信的一种非常常见的方式，通常用于调试。这是一种通信协议，其中数据通过开关输出引脚一次一个位来传输。我们可以使用它发送和接收任何内容，从原始二进制数据到文本和数字。

Arduino IDE 包含用于捕获和显示通过串行端口接收的数据的工具。其中一个工具是串行绘图器，可以显示通过串行接收的值的图形。通过从我们的代码输出一系列亮度值，我们将能够看到它们的图形。图 6-3 展示了这一过程。

![Arduino IDE 的串行绘图器截图](img/timl_0603.png)

###### 图 6-3\. Arduino IDE 的串行绘图器

我们将在本节后面提供如何使用串行绘图器的说明。

###### 注意

您可能想知道`ErrorReporter`如何通过 Arduino 的串行接口输出数据。您可以在[*micro/arduino/debug_log.cc*](https://oreil.ly/fkF8H)中找到代码实现。它替换了[*micro/debug_log.cc*](https://oreil.ly/nxXgJ)中的原始实现。就像*output_handler.cc*被覆盖一样，我们可以通过将它们添加到以平台名称命名的目录中，为 TensorFlow Lite for Microcontrollers 中的任何源文件提供特定于平台的实现。

## 运行示例

我们的下一个任务是为 Arduino 构建项目并将其部署到设备上。

###### 提示

建议检查[*README.md*](https://oreil.ly/s2mj1)以获取最新的指导，因为自本书编写以来构建过程可能已经发生变化。

我们需要的一切如下：

+   支持的 Arduino 板（我们推荐 Arduino Nano 33 BLE Sense）

+   适当的 USB 电缆

+   [Arduino IDE](https://oreil.ly/c-rv6)（您需要下载并安装此软件才能继续）

本书中的项目作为 TensorFlow Lite Arduino 库中的示例代码可用，您可以通过 Arduino IDE 轻松安装，并从工具菜单中选择管理库。在弹出的窗口中，搜索并安装名为*Arduino_TensorFlowLite*的库。您应该能够使用最新版本，但如果遇到问题，本书测试过的版本是`1.14-ALPHA`。

###### 注意

您还可以从*.zip*文件安装库，您可以从 TensorFlow Lite 团队[下载](https://oreil.ly/blgB8)，或者使用 TensorFlow Lite for Microcontrollers Makefile 自动生成。如果您更喜欢这样做，请参见附录 A。

安装完库后，`hello_world`示例将显示在文件菜单下的 Examples→Arduino_TensorFlowLite 中，如图 6-4 所示。

单击“hello_world”加载示例。它将显示为一个新窗口，每个源文件都有一个选项卡。第一个选项卡中的文件*hello_world*相当于我们之前讨论过的*main_functions.cc*。

![“示例”菜单的截图](img/timl_0604.png)

###### 图 6-4\. 示例菜单

要运行示例，请通过 USB 连接您的 Arduino 设备。确保在工具菜单中从板下拉列表中选择正确的设备类型，如图 6-5 所示。

![“板”下拉菜单的截图](img/timl_0605.png)

###### 图 6-5。板下拉列表

如果您的设备名称未出现在列表中，则需要安装其支持包。要执行此操作，请单击“Boards Manager”。在出现的窗口中，搜索您的设备并安装相应支持包的最新版本。

接下来，请确保在“端口”下拉列表中选择了设备的端口，也在工具菜单中，如图 6-6 所示。

![“端口”下拉列表的屏幕截图](img/timl_0606.png)

###### 图 6-6。端口下拉列表

最后，在 Arduino 窗口中，单击上传按钮（在图 6-7 中用白色突出显示）来编译并将代码上传到您的 Arduino 设备。

![具有箭头图标的上传按钮的屏幕截图](img/timl_0607.png)

###### 图 6-7。上传按钮，一个右箭头

上传成功完成后，您应该看到 Arduino 板上的 LED 开始淡入淡出或闪烁，具体取决于其连接的引脚是否支持 PWM。

恭喜：您正在设备上运行 ML！

###### 注意

不同型号的 Arduino 板具有不同的硬件，并且将以不同的速度运行推断。如果您的 LED 要么闪烁要么保持完全开启，您可能需要增加每个周期的推断次数。您可以通过*arduino_constants.cpp*中的`kInferencesPerCycle`常量来实现这一点。

“进行您自己的更改”向您展示如何编辑示例代码。

您还可以查看绘制在图表上的亮度值。要执行此操作，请在工具菜单中选择 Arduino IDE 的串行绘图器，如图 6-8 所示。

![串行绘图器菜单选项的屏幕截图](img/timl_0608.png)

###### 图 6-8。串行绘图器菜单选项

绘图器显示随时间变化的值，如图 6-9 所示。

![Arduino IDE 的串行图绘屏幕截图](img/timl_0603.png)

###### 图 6-9。串行绘图器绘制值

要查看从 Arduino 串行端口接收的原始数据，请从工具菜单中打开串行监视器。您将看到一系列数字飞过，就像图 6-10 中所示。

![Arduino IDE 的串行监视器的屏幕截图](img/timl_0610.png)

###### 图 6-10。显示原始数据的串行监视器

## 进行您自己的更改

现在您已经部署了应用程序，可能会很有趣地玩耍并对代码进行一些更改。您可以在 Arduino IDE 中编辑源文件。保存时，您将被提示在新位置重新保存示例。完成更改后，您可以在 Arduino IDE 中单击上传按钮来构建和部署。

要开始进行更改，您可以尝试以下几个实验：

+   通过调整每个周期的推断次数来使 LED 闪烁速度变慢或变快。

+   修改*output_handler.cc*以将基于文本的动画记录到串行端口。

+   使用正弦波来控制其他组件，如额外的 LED 或声音发生器。

# SparkFun Edge

[SparkFun Edge](https://oreil.ly/-hoL-)开发板专门设计为在微型设备上进行机器学习实验的平台。它具有功耗高效的 Ambiq Apollo 3 微控制器，带有 Arm Cortex M4 处理器核心。

它具有四个 LED 的一组，如图 6-11 所示。我们使用这些 LED 来直观输出我们的正弦值。

![突出显示其四个 LED 的 SparkFun Edge 的照片](img/timl_0611.png)

###### 图 6-11。SparkFun Edge 的四个 LED

## 在 SparkFun Edge 上处理输出

我们可以使用板上的一组 LED 制作一个简单的动画，因为没有什么比[blinkenlights](https://oreil.ly/T90fy)更能展示尖端人工智能了。

LED（红色、绿色、蓝色和黄色）在以下顺序中物理排列：

```py
                         [ R G B Y ]
```

以下表格表示我们将如何为不同的`y`值点亮 LED：

| 范围 | 点亮的 LED |
| --- | --- |
| `0.75 <= y <= 1` | `[ 0 0 1 1 ]` |
| `0 < y < 0.75` | `[ 0 0 1 0 ]` |
| `y = 0` | `[ 0 0 0 0 ]` |
| `-0.75 < y < 0` | `[ 0 1 0 0 ]` |
| `-1 <= y <= 0.75` | `[ 1 1 0 0 ]` |

每次推断需要一定的时间，所以调整`kInferencesPerCycle`，在*constants.cc*中定义，将调整 LED 循环的速度。

图 6-12 显示了程序运行的一个静态图像，来自一个动画*.gif*。

![来自 SparkFun Edge LED 动画的静态图像](img/timl_0612.png)

###### 图 6-12。来自 SparkFun Edge LED 动画的静态图像

实现 SparkFun Edge 的输出处理的代码在[*hello_world/sparkfun_edge/output_handler.cc*](https://oreil.ly/tegLK)中，用于替代原始文件*hello_world/output_handler.cc*。

让我们开始逐步进行：

```py
#include "tensorflow/lite/micro/examples/hello_world/output_handler.h"
#include "am_bsp.h"
```

首先，我们包含一些头文件。我们的*output_handler.h*指定了此文件的接口。另一个文件*am_bsp.h*来自一个叫做*Ambiq Apollo3 SDK*的东西。Ambiq 是 SparkFun Edge 微控制器的制造商，称为 Apollo3。SDK（软件开发工具包）是一组源文件，定义了可以用来控制微控制器功能的常量和函数。

因为我们计划控制板上的 LED，所以我们需要能够打开和关闭微控制器的引脚。这就是我们使用 SDK 的原因。

###### 注意

当我们最终构建项目时，Makefile 将自动下载 SDK。如果你感兴趣，可以在[SparkFun 的网站](https://oreil.ly/RHHqI)上阅读更多关于它的信息或下载代码进行探索。

接下来，我们定义`HandleOutput()`函数，并指示在其第一次运行时要执行的操作：

```py
void HandleOutput(tflite::ErrorReporter* error_reporter, float x_value,
                  float y_value) {
  // The first time this method runs, set up our LEDs correctly
  static bool is_initialized = false;
  if (!is_initialized) {
    // Set up LEDs as outputs
    am_hal_gpio_pinconfig(AM_BSP_GPIO_LED_RED, g_AM_HAL_GPIO_OUTPUT_12);
    am_hal_gpio_pinconfig(AM_BSP_GPIO_LED_BLUE, g_AM_HAL_GPIO_OUTPUT_12);
    am_hal_gpio_pinconfig(AM_BSP_GPIO_LED_GREEN, g_AM_HAL_GPIO_OUTPUT_12);
    am_hal_gpio_pinconfig(AM_BSP_GPIO_LED_YELLOW, g_AM_HAL_GPIO_OUTPUT_12);
    // Ensure all pins are cleared
    am_hal_gpio_output_clear(AM_BSP_GPIO_LED_RED);
    am_hal_gpio_output_clear(AM_BSP_GPIO_LED_BLUE);
    am_hal_gpio_output_clear(AM_BSP_GPIO_LED_GREEN);
    am_hal_gpio_output_clear(AM_BSP_GPIO_LED_YELLOW);
    is_initialized = true;
  }
```

哦，这是很多的设置！我们使用`am_hal_gpio_pinconfig()`函数，由*am_bsp.h*提供，来配置连接到板上内置 LED 的引脚，将它们设置为输出模式（由`g_AM_HAL_GPIO_OUTPUT_12`常量表示）。每个 LED 的引脚号由一个常量表示，比如`AM_BSP_GPIO_LED_RED`。

然后我们使用`am_hal_gpio_output_clear()`清除所有输出，以便所有 LED 都关闭。与 Arduino 实现一样，我们使用名为`is_initialized`的`static`变量，以确保此块中的代码仅运行一次。接下来，我们确定如果`y`值为负时应点亮哪些 LED：

```py
// Set the LEDs to represent negative values
if (y_value < 0) {
  // Clear unnecessary LEDs
  am_hal_gpio_output_clear(AM_BSP_GPIO_LED_GREEN);
  am_hal_gpio_output_clear(AM_BSP_GPIO_LED_YELLOW);
  // The blue LED is lit for all negative values
  am_hal_gpio_output_set(AM_BSP_GPIO_LED_BLUE);
  // The red LED is lit in only some cases
  if (y_value <= -0.75) {
    am_hal_gpio_output_set(AM_BSP_GPIO_LED_RED);
  } else {
    am_hal_gpio_output_clear(AM_BSP_GPIO_LED_RED);
  }
```

首先，如果`y`值刚刚变为负数，我们清除用于指示正值的两个 LED。接下来，我们调用`am_hal_gpio_output_set()`来打开蓝色 LED，如果值为负数，它将始终点亮。最后，如果值小于-0.75，我们打开红色 LED。否则，我们关闭它。

接下来，我们做同样的事情，但是对于`y`的正值：

```py
  // Set the LEDs to represent positive values
} else if (y_value > 0) {
  // Clear unnecessary LEDs
  am_hal_gpio_output_clear(AM_BSP_GPIO_LED_RED);
  am_hal_gpio_output_clear(AM_BSP_GPIO_LED_BLUE);
  // The green LED is lit for all positive values
  am_hal_gpio_output_set(AM_BSP_GPIO_LED_GREEN);
  // The yellow LED is lit in only some cases
  if (y_value >= 0.75) {
    am_hal_gpio_output_set(AM_BSP_GPIO_LED_YELLOW);
  } else {
    am_hal_gpio_output_clear(AM_BSP_GPIO_LED_YELLOW);
  }
}
```

LED 就是这样。我们最后要做的是将当前的输出值记录到串口上正在监听的人：

```py
// Log the current X and Y values
error_reporter->Report("x_value: %f, y_value: %f\n", x_value, y_value);
```

###### 注意

我们的`ErrorReporter`能够通过 SparkFun Edge 的串行接口输出数据，这是由于[*micro/sparkfun_edge/debug_log.cc*](https://oreil.ly/ufEv9)的自定义实现取代了[*mmicro/debug_log.cc*](https://oreil.ly/ACaFt)中的原始实现。

## 运行示例

现在我们可以构建示例代码并将其部署到 SparkFun Edge 上。

###### 提示

构建过程可能会有变化，因为这本书写作时，所以请查看[*README.md*](https://oreil.ly/EcPZ8)获取最新的指导。

要构建和部署我们的代码，我们需要以下内容：

+   一个 SparkFun Edge 板

+   一个 USB 编程器（我们推荐 SparkFun Serial Basic Breakout，可在[micro-B USB](https://oreil.ly/A6oDw)和[USB-C](https://oreil.ly/3REjg)变体中获得）

+   一根匹配的 USB 电缆

+   Python 3 和一些依赖项

首先，打开一个终端，克隆 TensorFlow 存储库，然后切换到其目录：

```py
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
```

接下来，我们将构建二进制文件，并运行一些命令，使其准备好下载到设备中。为了避免一些打字错误，您可以从[*README.md*](https://oreil.ly/PYmUu)中复制并粘贴这些命令。

### 构建二进制文件

以下命令下载所有必需的依赖项，然后为 SparkFun Edge 编译一个二进制文件：

```py
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=sparkfun_edge hello_world_bin
```

###### 注意

二进制文件是包含程序的文件，可以直接由 SparkFun Edge 硬件运行。

二进制文件将被创建为*.bin*文件，位置如下：

```py
tensorflow/lite/micro/tools/make/gen/ \
  sparkfun_edge_cortex-m4/bin/hello_world.bin
```

要检查文件是否存在，您可以使用以下命令：

```py
test -f tensorflow/lite/micro/tools/make/gen/ \
  sparkfun_edge_cortex-m4/bin/hello_world.bin \
  &&  echo "Binary was successfully created" || echo "Binary is missing"
```

如果运行该命令，您应该看到`二进制文件已成功创建`打印到控制台。

如果看到`二进制文件丢失`，则构建过程中出现问题。如果是这样，很可能您可以在`make`命令的输出中找到一些出错的线索。

### 对二进制文件进行签名

必须使用加密密钥对二进制文件进行签名，以便部署到设备上。现在让我们运行一些命令，对二进制文件进行签名，以便将其刷写到 SparkFun Edge。这里使用的脚本来自 Ambiq SDK，在运行 Makefile 时下载。

输入以下命令设置一些虚拟的加密密钥，供开发使用：

```py
cp tensorflow/lite/micro/tools/make/downloads/AmbiqSuite-Rel2.0.0/ \
  tools/apollo3_scripts/keys_info0.py \
  tensorflow/lite/micro/tools/make/downloads/AmbiqSuite-Rel2.0.0/ \
  tools/apollo3_scripts/keys_info.py
```

接下来，运行以下命令创建一个已签名的二进制文件。如有必要，将`python3`替换为`python`：

```py
python3 tensorflow/lite/micro/tools/make/downloads/ \
  AmbiqSuite-Rel2.0.0/tools/apollo3_scripts/create_cust_image_blob.py \
  --bin tensorflow/lite/micro/tools/make/gen/ \
  sparkfun_edge_cortex-m4/bin/hello_world.bin \
  --load-address 0xC000 \
  --magic-num 0xCB -o main_nonsecure_ota \
  --version 0x0
```

这将创建文件*main_nonsecure_ota.bin*。现在运行以下命令以创建文件的最终版本，您可以使用该文件通过下一步中将使用的脚本刷写设备：

```py
python3 tensorflow/lite/micro/tools/make/downloads/ \
  AmbiqSuite-Rel2.0.0/tools/apollo3_scripts/create_cust_wireupdate_blob.py \
  --load-address 0x20000 \
  --bin main_nonsecure_ota.bin \
  -i 6 \
  -o main_nonsecure_wire \
  --options 0x1
```

您现在应该在运行命令的目录中有一个名为*main_nonsecure_wire.bin*的文件。这是您将要刷写到设备上的文件。

### 刷写二进制文件

SparkFun Edge 将当前运行的程序存储在其 1 兆字节的闪存中。如果要让板运行新程序，您需要将其发送到板上，板将其存储在闪存中，覆盖先前保存的任何程序。

这个过程称为*刷写*。让我们一步步走过这些步骤。

#### 将编程器连接到板上

要将新程序下载到板上，您将使用 SparkFun USB-C 串行基本串行编程器。该设备允许您的计算机通过 USB 与微控制器通信。

要将此设备连接到您的板上，请执行以下步骤：

1.  在 SparkFun Edge 的一侧，找到六针排针。

1.  将 SparkFun USB-C 串行基本插入这些引脚，确保每个设备上标有 BLK 和 GRN 的引脚正确对齐。

您可以在图 6-13 中看到正确的排列方式。

![显示 SparkFun Edge 和 USB-C 串行基本连接方式的照片](img/timl_0613.png)

###### 图 6-13\. 连接 SparkFun Edge 和 USB-C 串行基本（由 SparkFun 提供）

#### 将编程器连接到计算机

接下来，通过 USB 将板连接到计算机。要对板进行编程，您需要确定计算机给设备的名称。最好的方法是在连接设备之前和之后列出计算机的所有设备，然后查看哪个设备是新的。

###### 警告

有些人报告了使用编程器的操作系统默认驱动程序出现问题，因此我们强烈建议在继续之前安装[驱动程序](https://oreil.ly/Wkxaf)。

在通过 USB 连接设备之前，请运行以下命令：

```py
# macOS:
ls /dev/cu*

# Linux:
ls /dev/tty*
```

这应该输出一个类似以下内容的附加设备列表：

```py
/dev/cu.Bluetooth-Incoming-Port
/dev/cu.MALS
/dev/cu.SOC
```

现在，将编程器连接到计算机的 USB 端口，并再次运行命令：

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

这个名称将用于引用设备。但是，它可能会根据编程器连接到的 USB 端口而改变，因此如果您从计算机断开板然后重新连接，可能需要再次查找其名称。

###### 提示

一些用户报告列表中出现了两个设备。如果看到两个设备，正确的设备名称应以“wch”开头；例如，“/dev/wchusbserial-14410”。

确定设备名称后，将其放入一个 shell 变量以供以后使用：

```py
export DEVICENAME=<*your device name here*>

```

这是一个变量，您可以在后续过程中运行需要设备名称的命令时使用。

#### 运行脚本以刷写您的板子

要刷写板子，您需要将其放入特殊的“引导加载程序”状态，以准备接收新的二进制文件。然后可以运行脚本将二进制文件发送到板子上。

首先创建一个环境变量来指定波特率，即数据发送到设备的速度：

```py
export BAUD_RATE=921600
```

现在将以下命令粘贴到您的终端中——但*不要按 Enter 键*！命令中的`${DEVICENAME}`和`${BAUD_RATE}`将被替换为您在前面部分设置的值。如果需要，请记得将`python3`替换为`python`：

```py
python3 tensorflow/lite/micro/tools/make/downloads/ \
  AmbiqSuite-Rel2.0.0/tools/apollo3_scripts/ \
  uart_wired_update.py -b ${BAUD_RATE} \
  ${DEVICENAME} -r 1 -f main_nonsecure_wire.bin -i 6
```

接下来，您将重置板子到引导加载程序状态并刷写板子。在板子上，找到标记为`RST`和`14`的按钮，如图 6-14 所示。

![显示 SparkFun Edge 按钮的照片](img/timl_0614.png)

###### 图 6-14。SparkFun Edge 的按钮

执行以下步骤：

1.  确保您的板子连接到编程器，并且整个设备通过 USB 连接到计算机。

1.  在板子上，按住标记为`14`的按钮。*继续按住它*。

1.  在继续按住标记为`14`的按钮的同时，按下标记为`RST`的按钮重置板子。

1.  在计算机上按 Enter 键运行脚本。*继续按住按钮`14`*。

现在您应该在屏幕上看到类似以下内容的东西：

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

继续按住按钮`14`，直到看到`发送数据包长度为 8180`。在看到这个之后可以释放按钮（但如果您继续按住也没关系）。

程序将继续在终端上打印行。最终您会看到类似以下内容的东西：

```py
[...lots more Sending Data Packet of length  8180...]
Sending Data Packet of length  8180
Sending Data Packet of length  6440
Sending Reset Command.
Done.
```

这表明刷写成功。

###### 提示

如果程序输出以错误结束，请检查是否打印了`发送复位命令`。如果是，则刷写可能成功，尽管有错误。否则，刷写可能失败。尝试再次运行这些步骤（您可以跳过设置环境变量）。

## 测试程序

现在应该已经将二进制文件部署到设备上。按下标记为`RST`的按钮重新启动板子。您应该看到设备的四个 LED 按顺序闪烁。干得好！

## 查看调试数据

在程序运行时，板子会记录调试信息。要查看它，我们可以使用波特率 115200 监视板子的串行端口输出。在 macOS 和 Linux 上，以下命令应该有效：

```py
screen ${DEVICENAME} 115200
```

您会看到大量的输出飞过！要停止滚动，请按下 Ctrl-A，紧接着按 Esc。然后您可以使用箭头键浏览输出，其中包含对各种`x`值运行推断的结果：

```py
x_value: 1.1843798*2², y_value: -1.9542645*2^-1
```

要停止使用`screen`查看调试输出，请按下 Ctrl-A，紧接着按下 K 键，然后按下 Y 键。

###### 注意

程序`screen`是一个有用的连接到其他计算机的实用程序程序。在这种情况下，我们使用它来监听 SparkFun Edge 板通过其串行端口记录的数据。如果您使用 Windows，可以尝试使用程序[`CoolTerm`](https://oreil.ly/sPWQP)来做同样的事情。

## 进行您自己的更改

现在您已经部署了基本应用程序，请尝试玩耍并进行一些更改。您可以在*tensorflow/lite/micro/examples/hello_world*文件夹中找到应用程序的代码。只需编辑并保存，然后重复之前的说明以将修改后的代码部署到设备上。

以下是您可以尝试的一些事情：

+   通过调整每个周期的推断次数使 LED 的闪烁速度变慢或变快。

+   修改*output_handler.cc*以将基于文本的动画记录到串行端口。

+   使用正弦波来控制其他组件，如额外的 LED 或声音发生器。

# ST Microelectronics STM32F746G Discovery Kit

[STM32F746G](https://oreil.ly/cvm4J)是一个带有相对强大的 Arm Cortex-M7 处理器核心的微控制器开发板。

这块板运行 Arm 的[Mbed OS](https://os.mbed.com)，这是一个专为构建和部署嵌入式应用程序而设计的嵌入式操作系统。这意味着我们可以使用本节中的许多指令来为其他 Mbed 设备构建。

STM32F746G 配备了一个附加的 LCD 屏幕，这将使我们能够构建一个更加复杂的视觉显示。

## 在 STM32F746G 上处理输出

现在我们有一个整个 LCD 可以玩耍，我们可以绘制一个漂亮的动画。让我们使用屏幕的*x*轴表示推断的数量，*y*轴表示我们预测的当前值。

我们将在这个值应该的地方绘制一个点，当我们循环遍历 0 到 2π的输入范围时，它将在屏幕上移动。图 6-15 展示了这个的线框图。

每个推断需要一定的时间，因此调整`kInferencesPerCycle`，在*constants.cc*中定义，将调整点的运动速度和平滑度。

图 6-16 显示了程序运行的[动画*.gif*](https://oreil.ly/1EM7C)的静止画面。

![我们将在 LCD 显示屏上绘制的动画的插图](img/timl_0615.png)

###### 图 6-15。我们将在 LCD 显示屏上绘制的动画

图 6-16 显示了程序运行的[动画*.gif*](https://oreil.ly/1EM7C)的静止画面。

![STM32F746G Discovery kit，带有 LCD 显示屏](img/timl_0616.png)

###### 图 6-16。在 STM32F746G Discovery kit 上运行的代码，带有 LCD 显示屏

实现 STM32F746G 的输出处理的代码位于[*hello_world/disco_f746ng/output_handler.cc*](https://oreil.ly/bj4iL)中，该文件用于替代原始文件*hello_world/output_handler.cc*。

让我们来看一下：

```py
#include "tensorflow/lite/micro/examples/hello_world/output_handler.h"
#include "LCD_DISCO_F746NG.h"
#include "tensorflow/lite/micro/examples/hello_world/constants.h"
```

首先，我们有一些头文件。我们的*output_handler.h*指定了这个文件的接口。由开发板制造商提供的*LCD_DISCO_F74NG.h*声明了我们将用来控制其 LCD 屏幕的接口。我们还包括*constants.h*，因为我们需要访问`kInferencesPerCycle`和`kXrange`。

接下来，我们设置了大量变量。首先是`LCD_DISCO_F746NG`的一个实例，它在*LCD_DISCO_F74NG.h*中定义，并提供了我们可以用来控制 LCD 的方法：

```py
// The LCD driver
LCD_DISCO_F746NG lcd;
```

有关`LCD_DISCO_F746NG`类的详细信息可在[Mbed 网站](https://oreil.ly/yiPHS)上找到。

接下来，我们定义一些控制视觉外观和感觉的常量：

```py
// The colors we'll draw
const uint32_t background_color = 0xFFF4B400;  // Yellow
const uint32_t foreground_color = 0xFFDB4437;  // Red
// The size of the dot we'll draw
const int dot_radius = 10;
```

颜色以十六进制值提供，如`0xFFF4B400`。它们的格式为`AARRGGBB`，其中`AA`表示 alpha 值（或不透明度，`FF`表示完全不透明），`RR`、`GG`和`BB`表示红色、绿色和蓝色的量。

###### 提示

通过练习，您可以学会从十六进制值中读取颜色。`0xFFF4B400`是完全不透明的，有很多红色和一定量的绿色，这使得它成为一个漂亮的橙黄色。

您也可以通过快速的谷歌搜索查找这些值。

然后我们声明了一些变量，定义了动画的形状和大小：

```py
// Size of the drawable area
int width;
int height;
// Midpoint of the y axis
int midpoint;
// Pixels per unit of x_value
int x_increment;
```

在变量之后，我们定义了`HandleOutput()`函数，并告诉它在第一次运行时要做什么：

```py
// Animates a dot across the screen to represent the current x and y values
void HandleOutput(tflite::ErrorReporter* error_reporter, float x_value,
                  float y_value) {
  // Track whether the function has run at least once
  static bool is_initialized = false;

  // Do this only once
  if (!is_initialized) {
    // Set the background and foreground colors
    lcd.Clear(background_color);
    lcd.SetTextColor(foreground_color);
    // Calculate the drawable area to avoid drawing off the edges
    width = lcd.GetXSize() - (dot_radius * 2);
    height = lcd.GetYSize() - (dot_radius * 2);
    // Calculate the y axis midpoint
    midpoint = height / 2;
    // Calculate fractional pixels per unit of x_value
    x_increment = static_cast<float>(width) / kXrange;
    is_initialized = true;
  }
```

里面有很多内容！首先，我们使用属于`lcd`的方法来设置背景和前景颜色。奇怪命名的`lcd.SetTextColor()`设置我们绘制的任何东西的颜色，不仅仅是文本：

```py
// Set the background and foreground colors
lcd.Clear(background_color);
lcd.SetTextColor(foreground_color);
```

接下来，我们计算实际可以绘制到屏幕的部分，以便知道在哪里绘制我们的圆。如果我们搞错了，我们可能会尝试绘制超出屏幕边缘，导致意想不到的结果：

```py
width = lcd.GetXSize() - (dot_radius * 2);
height = lcd.GetYSize() - (dot_radius * 2);
```

之后，我们确定屏幕中间的位置，我们将在其下方绘制负`y`值。我们还计算屏幕宽度中表示一个单位`x`值的像素数。请注意我们如何使用`static_cast`确保获得浮点结果：

```py
// Calculate the y axis midpoint
midpoint = height / 2;
// Calculate fractional pixels per unit of x_value
x_increment = static_cast<float>(width) / kXrange;
```

与之前一样，使用名为`is_initialized`的`static`变量确保此块中的代码仅运行一次。

初始化完成后，我们可以开始输出。首先，清除任何先前的绘图：

```py
// Clear the previous drawing
lcd.Clear(background_color);
```

接下来，我们使用`x_value`来计算我们应该在显示器的*x*轴上绘制点的位置：

```py
// Calculate x position, ensuring the dot is not partially offscreen,
// which causes artifacts and crashes
int x_pos = dot_radius + static_cast<int>(x_value * x_increment);
```

然后我们对`y`值执行相同的操作。这有点复杂，因为我们希望在`midpoint`上方绘制正值，在下方绘制负值：

```py
// Calculate y position, ensuring the dot is not partially offscreen
int y_pos;
if (y_value >= 0) {
  // Since the display's y runs from the top down, invert y_value
  y_pos = dot_radius + static_cast<int>(midpoint * (1.f - y_value));
} else {
  // For any negative y_value, start drawing from the midpoint
  y_pos =
      dot_radius + midpoint + static_cast<int>(midpoint * (0.f - y_value));
}
```

一旦确定了它的位置，我们就可以继续绘制点：

```py
// Draw the dot
lcd.FillCircle(x_pos, y_pos, dot_radius);
```

最后，我们使用我们的`ErrorReporter`将`x`和`y`值记录到串行端口：

```py
// Log the current X and Y values
error_reporter->Report("x_value: %f, y_value: %f\n", x_value, y_value);
```

###### 注意

由于自定义实现，`ErrorReporter`可以通过 STM32F746G 的串行接口输出数据，[*micro/disco_f746ng/debug_log.cc*](https://oreil.ly/eL1ft)，取代了[*micro/debug_log.cc*](https://oreil.ly/HpJ-t)中的原始实现。

## 运行示例

接下来，让我们构建项目！STM32F746G 运行 Arm 的 Mbed OS，因此我们将使用 Mbed 工具链将我们的应用程序部署到设备上。

###### 提示

建议检查[*README.md*](https://oreil.ly/WuhIz)以获取最新说明，因为构建过程可能会有所变化。

在开始之前，我们需要以下内容：

+   一个 STM32F746G Discovery kit 板

+   一个迷你 USB 电缆

+   Arm Mbed CLI（请参阅[Mbed 设置指南](https://oreil.ly/TkRwd)）

+   Python 3 和`pip`

与 Arduino IDE 类似，Mbed 要求源文件以特定方式结构化。 TensorFlow Lite for Microcontrollers Makefile 知道如何为我们做到这一点，并且可以生成适用于 Mbed 的目录。

为了这样做，请运行以下命令：

```py
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=mbed TAGS="CMSIS disco_f746ng" generate_hello_world_mbed_project
```

这将导致创建一个新目录：

```py
tensorflow/lite/micro/tools/make/gen/mbed_cortex-m4/prj/ \
  hello_world/mbed
```

该目录包含所有示例的依赖项，以适合 Mbed 构建。

首先，切换到目录，以便您可以在其中运行一些命令：

```py
cd tensorflow/lite/micro/tools/make/gen/mbed_cortex-m4/prj/ \
  hello_world/mbed
```

现在，您将使用 Mbed 下载依赖项并构建项目。

要开始，请使用以下命令指定 Mbed 当前目录是 Mbed 项目的根目录：

```py
mbed config root .
```

接下来，指示 Mbed 下载依赖项并准备构建：

```py
mbed deploy
```

默认情况下，Mbed 将使用 C++98 构建项目。但是，TensorFlow Lite 需要 C++11。运行以下 Python 片段修改 Mbed 配置文件，以便使用 C++11。您可以直接在命令行中键入或粘贴：

```py
python -c 'import fileinput, glob;
for filename in glob.glob("mbed-os/tools/profiles/*.json"):
  for line in fileinput.input(filename, inplace=True):
    print(line.replace("\"-std=gnu++98\"","\"-std=c++11\", \"-fpermissive\""))'
```

最后，运行以下命令进行编译：

```py
mbed compile -m DISCO_F746NG -t GCC_ARM
```

这应该会在以下路径生成一个二进制文件：

```py
cp ./BUILD/DISCO_F746NG/GCC_ARM/mbed.bin
```

使用 Mbed 启动的一个好处是，像 STM32F746G 这样的 Mbed 启用板的部署非常简单。要部署，只需将 STM 板插入并将文件复制到其中。在 macOS 上，您可以使用以下命令执行此操作：

```py
cp ./BUILD/DISCO_F746NG/GCC_ARM/mbed.bin /Volumes/DIS_F746NG/
```

或者，只需在文件浏览器中找到`DIS_F746NG`卷并将文件拖放过去。复制文件将启动闪存过程。完成后，您应该在设备屏幕上看到动画。

除了这个动画之外，当程序运行时，板上还会记录调试信息。要查看它，请使用波特率为 9600 的串行连接与板建立串行连接。

在 macOS 和 Linux 上，当您发出以下命令时，设备应该会列出：

```py
ls /dev/tty*
```

它看起来会像下面这样：

```py
/dev/tty.usbmodem1454203
```

确定设备后，请使用以下命令连接到设备，将<*`/dev/tty.devicename`*>替换为设备名称，该名称显示在*/dev*中：

```py
screen /<*dev/tty.devicename*> 9600

```

您会看到很多输出飞过。要停止滚动，请按 Ctrl-A，然后立即按 Esc。然后，您可以使用箭头键浏览输出，其中包含在各种`x`值上运行推断的结果：

```py
x_value: 1.1843798*2², y_value: -1.9542645*2^-1
```

要停止使用`screen`查看调试输出，请按 Ctrl-A，紧接着按 K 键，然后按 Y 键。

## 进行您自己的更改

现在您已经部署了应用程序，可以尝试玩耍并进行一些更改！您可以在*tensorflow/lite/micro/tools/make/gen/mbed_cortex-m4/prj/hello_world/mbed*文件夹中找到应用程序的代码。只需编辑并保存，然后重复之前的说明以将修改后的代码部署到设备上。

以下是您可以尝试的一些事情：

+   通过调整每个周期的推理次数来使点移动更慢或更快。

+   修改*output_handler.cc*以将基于文本的动画记录到串行端口。

+   使用正弦波来控制其他组件，比如 LED 或声音发生器。

# 总结

在过去的三章中，我们已经经历了训练模型、将其转换为 TensorFlow Lite、围绕其编写应用程序并将其部署到微型设备的完整端到端旅程。在接下来的章节中，我们将探索一些更复杂和令人兴奋的示例，将嵌入式机器学习投入实际应用。

首先，我们将构建一个使用微小的、18 KB 模型识别口头命令的应用程序。
