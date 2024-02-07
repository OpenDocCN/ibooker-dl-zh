# 附录B. 在Arduino上捕获音频

以下文本将从[第7章](ch07.xhtml#chapter_speech_wake_word_example)中唤醒词应用程序的音频捕获代码中走过。由于它与机器学习没有直接关系，所以作为附录提供。

Arduino Nano 33 BLE Sense具有内置麦克风。要从麦克风接收音频数据，我们可以注册一个回调函数，当有一块新的音频数据准备好时就会调用它。

每次发生这种情况，我们将新数据块写入存储数据储备的*缓冲区*中。由于音频数据占用大量内存，缓冲区只能容纳一定量的数据。当缓冲区变满时，这些数据将被覆盖。

每当我们的程序准备好运行推断时，它可以从该缓冲区中读取最近一秒钟的数据。只要新数据持续进入得比我们需要访问的快，缓冲区中总是会有足够的新数据进行预处理并馈入我们的模型。

每个预处理和推断周期都很复杂，需要一些时间来完成。因此，在Arduino上我们每秒只能运行几次推断。这意味着我们的缓冲区很容易保持满状态。

正如我们在[第7章](ch07.xhtml#chapter_speech_wake_word_example)中看到的，*audio_provider.h*实现了这两个函数：

+   `GetAudioSamples()`，提供指向一块原始音频数据的指针

+   `LatestAudioTimestamp()`，返回最近捕获音频的时间戳

实现这些功能的Arduino代码位于[*arduino/audio_provider.cc*](https://oreil.ly/Bfh4v)中。

在第一部分中，我们引入了一些依赖项。*PDM.h*库定义了我们将用来从麦克风获取数据的API。文件*micro_model_settings.h*包含了与我们模型数据需求相关的常量，这将帮助我们以正确的格式提供音频数据。

```py
#include "tensorflow/lite/micro/examples/micro_speech/
  audio_provider.h"

#include "PDM.h"
#include "tensorflow/lite/micro/examples/micro_speech/
  micro_features/micro_model_settings.h"
```

接下来的代码块是设置一些重要变量的地方：

```py
namespace {
bool g_is_audio_initialized = false;
// An internal buffer able to fit 16x our sample size
constexpr int kAudioCaptureBufferSize = DEFAULT_PDM_BUFFER_SIZE * 16;
int16_t g_audio_capture_buffer[kAudioCaptureBufferSize];
// A buffer that holds our output
int16_t g_audio_output_buffer[kMaxAudioSampleSize];
// Mark as volatile so we can check in a while loop to see if
// any samples have arrived yet.
volatile int32_t g_latest_audio_timestamp = 0;
}  // namespace
```

布尔值`g_is_audio_initialized`用于跟踪麦克风是否已开始捕获音频。我们的音频捕获缓冲区由`g_audio_capture_buffer`定义，大小为`DEFAULT_PDM_BUFFER_SIZE`的16倍，这是在*PDM.h*中定义的一个常量，表示每次调用回调函数时从麦克风接收的音频量。拥有一个很大的缓冲区意味着如果程序因某种原因变慢，我们不太可能耗尽数据。

除了音频捕获缓冲区，我们还保留一个输出音频缓冲区`g_audio_output_buffer`，当调用`GetAudioSamples()`时，我们将返回一个指向它的指针。它的长度是`kMaxAudioSampleSize`，这是来自*micro_model_settings.h*的一个常量，定义了我们的预处理代码一次可以处理的16位音频样本的数量。

最后，我们使用`g_latest_audio_timestamp`来跟踪我们最新音频样本所代表的时间。这不会与您手表上的时间匹配；它只是相对于音频捕获开始时的毫秒数。该变量声明为`volatile`，这意味着处理器不应尝试缓存其值。稍后我们会看到原因。

设置这些变量后，我们定义回调函数，每当有新的音频数据可用时就会调用它。以下是完整的函数：

```py
void CaptureSamples() {
  // This is how many bytes of new data we have each time this is called
  const int number_of_samples = DEFAULT_PDM_BUFFER_SIZE;
  // Calculate what timestamp the last audio sample represents
  const int32_t time_in_ms =
      g_latest_audio_timestamp +
      (number_of_samples / (kAudioSampleFrequency / 1000));
  // Determine the index, in the history of all samples, of the last sample
  const int32_t start_sample_offset =
      g_latest_audio_timestamp * (kAudioSampleFrequency / 1000);
  // Determine the index of this sample in our ring buffer
  const int capture_index = start_sample_offset % kAudioCaptureBufferSize;
  // Read the data to the correct place in our buffer
  PDM.read(g_audio_capture_buffer + capture_index, DEFAULT_PDM_BUFFER_SIZE);
  // This is how we let the outside world know that new audio data has arrived.
  g_latest_audio_timestamp = time_in_ms;
}
```

这个函数有点复杂，所以我们将分块解释。它的目标是确定正确的索引，将这些新数据写入音频捕获缓冲区。

首先，我们确定每次调用回调函数时将接收多少新数据。我们使用这个数据来确定一个以毫秒表示缓冲区中最近音频样本的时间的数字：

```py
// This is how many bytes of new data we have each time this is called
const int number_of_samples = DEFAULT_PDM_BUFFER_SIZE;
// Calculate what timestamp the last audio sample represents
const int32_t time_in_ms =
    g_latest_audio_timestamp +
    (number_of_samples / (kAudioSampleFrequency / 1000));
```

每秒的音频样本数是`kAudioSampleFrequency`（这个常量在*micro_model_settings.h*中定义）。我们将这个数除以1,000得到每毫秒的样本数。

接下来，我们将每个回调的样本数（`number_of_samples`）除以每毫秒的样本数以获取每个回调获得的数据的毫秒数：

```py
(number_of_samples / (kAudioSampleFrequency / 1000))
```

然后，我们将其添加到我们先前最近音频样本的时间戳`g_latest_audio_timestamp`，以获取最新新音频样本的时间戳。

当我们有了这个数字后，我们可以使用它来获取*所有样本历史记录*中最近样本的索引。为此，我们将先前最近音频样本的时间戳乘以每毫秒的样本数：

```py
const int32_t start_sample_offset =
    g_latest_audio_timestamp * (kAudioSampleFrequency / 1000);
```

然而，我们的缓冲区没有足够的空间来存储每个捕获的样本。相反，它有16倍`DEFAULT_PDM_BUFFER_SIZE`的空间。一旦数据超过这个限制，我们就开始用新数据覆盖缓冲区。

现在，我们有了我们新样本在*所有样本历史记录*中的索引。接下来，我们需要将其转换为实际缓冲区内样本的正确索引。为此，我们可以通过缓冲区长度除以历史索引并获取余数。这是使用模运算符（`%`）完成的：

```py
// Determine the index of this sample in our ring buffer
const int capture_index = start_sample_offset % kAudioCaptureBufferSize;
```

因为缓冲区的大小`kAudioCaptureBufferSize`是`DEFAULT_PDM_BUFFER_SIZE`的倍数，新数据将始终完全适合缓冲区。模运算符将返回新数据应开始的缓冲区内的索引。

接下来，我们使用`PDM.read()`方法将最新音频读入音频捕获缓冲区：

```py
// Read the data to the correct place in our buffer
PDM.read(g_audio_capture_buffer + capture_index, DEFAULT_PDM_BUFFER_SIZE);
```

第一个参数接受一个指向数据应写入的内存位置的指针。变量`g_audio_capture_buffer`是指向音频捕获缓冲区起始地址的指针。通过将`capture_index`添加到此位置，我们可以计算出要写入新数据的内存中的正确位置。第二个参数定义应读取多少数据，我们选择最大值`DEFAULT_PDM_BUFFER_SIZE`。

最后，我们更新`g_latest_audio_timestamp`：

```py
// This is how we let the outside world know that new audio data has arrived.
g_latest_audio_timestamp = time_in_ms;
```

这将通过`LatestAudioTimestamp()`方法暴露给程序的其他部分，让它们知道何时有新数据可用。因为`g_latest_audio_timestamp`声明为`volatile`，每次访问时其值将从内存中查找。这很重要，否则变量将被处理器缓存。因为其值在回调中设置，处理器不会知道要刷新缓存的值，任何访问它的代码都不会收到其当前值。

您可能想知道是什么使`CaptureSamples()`充当回调函数。它如何知道何时有新音频可用？这是我们代码的下一部分处理的，这部分是启动音频捕获的函数：

```py
TfLiteStatus InitAudioRecording(tflite::ErrorReporter* error_reporter) {
  // Hook up the callback that will be called with each sample
  PDM.onReceive(CaptureSamples);
  // Start listening for audio: MONO @ 16KHz with gain at 20
  PDM.begin(1, kAudioSampleFrequency);
  PDM.setGain(20);
  // Block until we have our first audio sample
  while (!g_latest_audio_timestamp) {
  }

  return kTfLiteOk;
}
```

第一次有人调用`GetAudioSamples()`时将调用此函数。它首先使用`PDM`库通过调用`PDM.onReceive()`来连接`CaptureSamples()`回调。接下来，使用两个参数调用`PDM.begin()`。第一个参数指示要记录多少个音频通道；我们只需要单声道音频，因此指定`1`。第二个参数指定每秒要接收多少样本。

接下来，使用`PDM.setGain()`来配置*增益*，定义麦克风音频应放大多少。我们指定增益为`20`，这是在一些实验之后选择的。

最后，我们循环直到`g_latest_audio_timestamp`评估为true。因为它从`0`开始，这会阻止执行，直到回调捕获到一些音频，此时`g_latest_audio_timestamp`将具有非零值。

我们刚刚探讨的两个函数允许我们启动捕获音频的过程并将捕获的音频存储在缓冲区中。接下来的函数`GetAudioSamples()`为我们代码的其他部分（即特征提供者）提供了获取音频数据的机制：

```py
TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
  // Set everything up to start receiving audio
  if (!g_is_audio_initialized) {
    TfLiteStatus init_status = InitAudioRecording(error_reporter);
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    g_is_audio_initialized = true;
  }
```

该函数被调用时带有一个用于写日志的`ErrorReporter`，两个指定我们请求的音频的变量（`start_ms`和`duration_ms`），以及用于传回音频数据的两个指针（`audio_samples_size`和`audio_samples`）。函数的第一部分调用`InitAudioRecording()`。正如我们之前看到的，这会阻塞执行，直到音频的第一个样本到达。我们使用变量`g_is_audio_initialized`来确保这个设置代码只运行一次。

在这一点之后，我们可以假设捕获缓冲区中存储了一些音频。我们的任务是找出正确音频数据在缓冲区中的位置。为了做到这一点，我们首先确定我们想要的第一个样本在*所有样本历史*中的索引：

```py
const int start_offset = start_ms * (kAudioSampleFrequency / 1000);
```

接下来，我们确定我们想要抓取的样本总数：

```py
const int duration_sample_count =
    duration_ms * (kAudioSampleFrequency / 1000);
```

现在我们有了这些信息，我们可以确定在我们的音频捕获缓冲区中从哪里读取。我们将在循环中读取数据：

```py
for (int i = 0; i < duration_sample_count; ++i) {
  // For each sample, transform its index in the history of all samples into
  // its index in g_audio_capture_buffer
  const int capture_index = (start_offset + i) % kAudioCaptureBufferSize;
  // Write the sample to the output buffer
  g_audio_output_buffer[i] = g_audio_capture_buffer[capture_index];
}
```

之前，我们看到如何使用取模运算符来找到在缓冲区内的正确位置，该缓冲区只有足够空间来容纳最近的样本。在这里，我们再次使用相同的技术——如果我们将当前索引*在所有样本历史中*除以音频捕获缓冲区的大小`kAudioCaptureBufferSize`，余数将指示数据在缓冲区中的位置。然后我们可以使用简单的赋值将数据从捕获缓冲区读取到输出缓冲区。

接下来，为了从这个函数中获取数据，我们使用作为参数提供的两个指针。它们分别是`audio_samples_size`，指向音频样本的数量，和`audio_samples`，指向输出缓冲区：

```py
  // Set pointers to provide access to the audio
  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_audio_output_buffer;

  return kTfLiteOk;
}
```

最后，我们通过返回`kTfLiteOk`来结束函数，让调用者知道操作成功了。

然后，在最后部分，我们定义`LatestAudioTimestamp()`：

```py
int32_t LatestAudioTimestamp() { return g_latest_audio_timestamp; }
```

由于这总是返回最近音频的时间戳，其他部分的代码可以在循环中检查它，以确定是否有新的音频数据到达。

这就是我们的音频提供程序的全部内容！我们现在确保我们的特征提供程序有稳定的新鲜音频样本供应。
