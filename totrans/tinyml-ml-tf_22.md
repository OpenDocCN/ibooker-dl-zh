# 附录 A：使用和生成 Arduino 库 Zip

Arduino IDE 要求源文件以一定的方式打包。TensorFlow Lite for Microcontrollers Makefile 知道如何为您做这件事，并且可以生成一个包含所有源文件的*.zip*文件，您可以将其导入到 Arduino IDE 作为库。这将允许您构建和部署您的应用程序。

在本节的后面会有生成此文件的说明。然而，开始的最简单方法是使用 TensorFlow 团队每晚生成的[预构建*.zip*文件](https://oreil.ly/blgB8)。

在下载了该文件之后，您需要导入它。在 Arduino IDE 的 Sketch 菜单中，选择包含库→添加.ZIP 库，如图 A-1 所示。

![“添加.ZIP 库…”菜单选项的屏幕截图](img/timl_aa01.png)

###### 图 A-1. “添加.ZIP 库…”菜单选项

在出现的文件浏览器中，找到*.zip*文件，然后点击选择以导入它。

您可能希望自己生成库，例如，如果您对 TensorFlow Git 存储库中的代码进行了更改，并希望在 Arduino 环境中测试这些更改。

如果您需要自己生成文件，请打开终端窗口，克隆 TensorFlow 存储库，并切换到其目录：

```py
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
```

现在运行以下脚本以生成*.zip*文件：

```py
tensorflow/lite/micro/tools/ci_build/test_arduino.sh
```

文件将被创建在以下位置：

```py
tensorflow/lite/micro/tools/make/gen/arduino_x86_64/ \
  prj/micro_speech/tensorflow_lite.zip
```

然后，您可以按照之前记录的步骤将此*.zip*文件导入到 Arduino IDE 中。如果您之前安装了库，您需要先删除原始版本。您可以通过从 Arduino IDE 的*libraries*目录中删除*tensorflow_lite*目录来实现这一点，您可以在 IDE 的首选项窗口中的“Sketchbook 位置”下找到它。
