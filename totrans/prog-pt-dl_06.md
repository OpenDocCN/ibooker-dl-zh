# 第六章. 声音之旅

深度学习最成功的应用之一是我们每天随身携带的东西。无论是 Siri 还是 Google Now，驱动这两个系统以及亚马逊的 Alexa 的引擎都是神经网络。在本章中，我们将看一下 PyTorch 的`torchaudio`库。您将学习如何使用它来构建一个用于分类音频数据的基于卷积的模型的流水线。之后，我将建议一种不同的方法，让您可以使用一些您学到的图像技巧，并在 ESC-50 音频数据集上获得良好的准确性。

但首先，让我们看看声音本身。它是什么？它通常以数据形式表示，这是否为我们提供了任何线索，告诉我们应该使用什么类型的神经网络来从数据中获得洞察？

# 声音

声音是通过空气的振动产生的。我们听到的所有声音都是高低压力的组合，我们通常用波形来表示，就像图 6-1 中的那个。在这个图像中，原点上方的波是高压，下方的部分是低压。

![正弦波](img/ppdl_0601.png)

###### 图 6-1. 正弦波

图 6-2 显示了一首完整歌曲的更复杂的波形。

![歌曲波形](img/ppdl_0602.png)

###### 图 6-2. 歌曲波形

在数字声音中，我们以每秒多次对这个波形进行*采样*，传统上是 44,100 次，用于 CD 音质，然后存储每个采样点的波的振幅值。在时间*t*，我们有一个单一的存储值。这与图像略有不同，图像需要两个值*x*和*y*来存储值（对于灰度图像）。如果我们在神经网络中使用卷积滤波器，我们需要一个 1D 滤波器，而不是我们用于图像的 2D 滤波器。

现在您对声音有了一些了解，让我们看看我们使用的数据集，这样您就可以更加熟悉它。

# ESC-50 数据集

*环境声音分类*（ESC）数据集是一组现场录音，每个录音长达 5 秒，并分配给 50 个类别之一（例如，狗叫、打鼾、敲门声）。我们将在本章的其余部分中使用此集合，以尝试两种分类音频的方法，并探索使用`torchaudio`来简化加载和操作音频。

## 获取数据集

[ESC-50 数据集](https://github.com/karoldvl/ESC-50)是一组 WAV 文件。您可以通过克隆 Git 存储库来下载它：

```py
git clone https://github.com/karoldvl/ESC-50
```

或者您可以使用 curl 下载整个存储库：

```py
curl https://github.com/karoldvl/ESC-50/archive/master.zip
```

所有的 WAV 文件都存储在*audio*目录中，文件名如下：

```py
1-100032-A-0.wav
```

我们关心文件名中的最后一个数字，因为这告诉我们这个声音片段被分配到了哪个类别。文件名的其他部分对我们来说并不重要，但大多数与 ESC-50 所绘制的更大的 Freesound 数据集相关（有一个例外，我马上会回来解释）。如果您想了解更多信息，ESC-50 存储库中的*README*文档会提供更详细的信息。

现在我们已经下载了数据集，让我们看看它包含的一些声音。

## 在 Jupyter 中播放音频

如果您想真正听到 ESC-50 中的声音，那么您可以使用 Jupyter 内置的音频播放器`IPython.display.Audio`，而不是将文件加载到标准音乐播放器（如 iTunes）中：

```py
import IPython.display as display
display.Audio('ESC-50/audio/1-100032-A-0.wav')
```

该函数将读取我们的 WAV 文件和 MP3 文件。您还可以生成张量，将它们转换为 NumPy 数组，并直接播放这些数组。播放*ESC-50*目录中的一些文件，以了解可用的声音。完成后，我们将更深入地探索数据集。

# 探索 ESC-50

处理新数据集时，最好在构建模型之前先了解数据的*形状*。例如，在分类任务中，你会想知道你的数据集是否实际包含了所有可能的类别的示例，并且最好所有类别的数量是相等的。让我们看看 ESC-50 是如何分解的。

###### 注意

如果你的数据集的数据量是*不平衡*的，一个简单的解决方案是随机复制较小类别的示例，直到你将它们增加到其他类别的数量。虽然这感觉像是虚假的账务，但在实践中它是令人惊讶地有效（而且便宜！）。

我们知道每个文件名中最后一组数字描述了它所属的类别，所以我们需要做的是获取文件列表并计算每个类别的出现次数：

```py
import glob
from collections import Counter

esc50_list = [f.split("-")[-1].replace(".wav","")
        for f in
        glob.glob("ESC-50/audio/*.wav")]
Counter(esc50_list)
```

首先，我们建立一个 ESC-50 文件名列表。因为我们只关心文件名末尾的类别编号，我们去掉 *.wav* 扩展名，并在 `-` 分隔符上分割文件名。最后我们取分割字符串中的最后一个元素。如果你检查 `esc50_list`，你会得到一堆从 0 到 49 的字符串。我们可以编写更多的代码来构建一个 `dict` 并为我们计算所有出现的次数，但我懒，所以我使用了一个 Python 的便利函数 `Counter`，它可以为我们做所有这些。

这是输出！

```py
Counter({'15': 40,
     '22': 40,
     '36': 40,
     '44': 40,
     '23': 40,
     '31': 40,
     '9': 40,
     '13': 40,
     '4': 40,
     '3': 40,
     '27': 40,
     …})
```

我们有一种罕见的完全平衡的数据集。让我们拿出香槟，安装一些我们很快会需要的库。

## SoX 和 LibROSA

`torchaudio` 进行的大部分音频处理依赖于另外两个软件：*SoX* 和 *LibROSA*。[*LibROSA*](https://github.com/librosa/librosa) 是一个用于音频分析的 Python 库，包括生成梅尔频谱图（你将在本章稍后看到这些），检测节拍，甚至生成音乐。

另一方面，*SoX* 是一个你可能已经熟悉的程序，如果你多年来一直在使用 Linux 的话。事实上，*SoX* 是如此古老，以至于它早于 Linux 本身；它的第一个版本是在 1991 年 7 月发布的，而 Linux 的首次亮相是在 1991 年 9 月。我记得在 1997 年使用它将 WAV 文件转换为 MP3 文件在我的第一台 Linux 电脑上。但它仍然很有用！

如果你通过 `conda` 安装 `torchaudio`，你可以跳到下一节。如果你使用 `pip`，你可能需要安装 *SoX* 本身。对于基于 Red Hat 的系统，输入以下命令：

```py
yum install sox
```

或者在基于 Debian 的系统上，你将使用以下命令：

```py
apt intall sox
```

安装 *SoX* 后，你可以继续获取 `torchaudio` 本身。

## torchaudio

安装 `torchaudio` 可以通过 `conda` 或 `pip` 进行：

```py
conda install -c derickl torchaudio
pip install torchaudio
```

与 `torchvision` 相比，`torchaudio` 类似于 `torchtext`，因为它并不像 `torchvision` 那样受到喜爱、维护或文档化。我预计随着 PyTorch 变得更受欢迎，更好的文本和音频处理流程将被创建，这种情况将在不久的将来发生改变。不过，`torchaudio` 对我们的需求来说已经足够了；我们只需要编写一些自定义的数据加载器（对于音频或文本处理，我们不需要这样做）。

无论如何，`torchaudio` 的核心在于 `load()` 和 `save()`。在本章中，我们只关心 `load()`，但如果你从输入生成新的音频（例如文本到语音模型），你需要使用 `save()`。`load()` 接受在 `filepath` 中指定的文件，并返回音频文件的张量表示和该音频文件的采样率作为一个单独的变量。

我们现在有了从 ESC-50 数据集中加载一个 WAV 文件并将其转换为张量的方法。与我们之前处理文本和图像的工作不同，我们需要写更多的代码才能继续创建和训练模型。我们需要编写一个自定义的 *dataset*。

## 构建 ESC-50 数据集

我们在第二章中讨论过数据集，但`torchvision`和`torchtext`为我们做了所有繁重的工作，所以我们不必太担心细节。你可能还记得，自定义数据集必须实现两个类方法，`__getitem__`和`__len__`，以便数据加载器可以获取一批张量及其标签，以及数据集中张量的总数。我们还有一个`__init__`方法用于设置诸如文件路径之类的东西，这些东西将一遍又一遍地使用。

这是我们对 ESC-50 数据集的第一次尝试：

```py
class ESC50(Dataset):

    def __init__(self,path):
        # Get directory listing from path
        files = Path(path).glob('*.wav')
        # Iterate through the listing and create a list of tuples (filename, label)
        self.items = [(f,int(f.name.split("-")[-1]
                    .replace(".wav",""))) for f in files]
        self.length = len(self.items)

    def __getitem__(self, index):
        filename, label = self.items[index]
        audio_tensor, sample_rate = torchaudio.load(filename)
        return audio_tensor, label

    def __len__(self):
        return self.length
```

类中的大部分工作发生在创建其新实例时。`__init__`方法接受`path`参数，找到该路径内的所有 WAV 文件，然后通过使用我们在本章早些时候使用的相同字符串拆分来生成*`(filename, label)`*元组，以获取该音频样本的标签。当 PyTorch 从数据集请求项目时，我们索引到`items`列表，使用`torchaudio.load`使`torchaudio`加载音频文件，将其转换为张量，然后返回张量和标签。

这就足够让我们开始了。为了进行健全性检查，让我们创建一个`ESC50`对象并提取第一个项目：

```py
test_esc50 = ESC50(PATH_TO_ESC50)
tensor, label = list(test_esc50)[0]

tensor
tensor([-0.0128, -0.0131, -0.0143,  ...,  0.0000,  0.0000,  0.0000])

tensor.shape
torch.Size([220500])

label
'15'
```

我们可以使用标准的 PyTorch 构造来构建数据加载器：

```py
example_loader = torch.utils.data.DataLoader(test_esc50, batch_size = 64,
shuffle = True)
```

但在这之前，我们必须回到我们的数据。您可能还记得，我们应该始终创建训练、验证和测试集。目前，我们只有一个包含所有数据的目录，这对我们的目的来说不好。将数据按 60/20/20 的比例分成训练、验证和测试集应该足够了。现在，我们可以通过随机抽取整个数据集的样本来做到这一点（注意要进行无重复抽样，并确保我们新构建的数据集仍然是平衡的），但是 ESC-50 数据集再次帮助我们省去了很多工作。数据集的编译者将数据分成了五个相等的平衡*folds*，文件名中的*第一个*数字表示。我们将`1,2,3`折作为训练集，`4`折作为验证集，`5`折作为测试集。但如果你不想无聊和连续，可以随意混合！将每个折叠移到*test*、*train*和*validation*目录中：

```py
mv 1* ../train
mv 2* ../train
mv 3* ../train
mv 4* ../valid
mv 5* ../test
```

现在我们可以创建各个数据集和加载器：

```py
from pathlib import Path

bs=64
PATH_TO_ESC50 = Path.cwd() / 'esc50'
path =  'test.md'
test

train_esc50 = ESC50(PATH_TO_ESC50 / "train")
valid_esc50 = ESC50(PATH_TO_ESC50 / "valid")
test_esc50  = ESC50(PATH_TO_ESC50 / "test")

train_loader = torch.utils.data.DataLoader(train_esc50, batch_size = bs,
                shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_esc50, batch_size = bs,
                shuffle = True)
test_loader  = torch.utils.data.DataLoader(test_esc50, batch_size = bs,
                shuffle = True)
```

我们已经准备好了我们的数据，所以我们现在准备好查看分类模型了。

# ESC-50 的 CNN 模型

对于我们第一次尝试分类声音，我们构建了一个模型，它大量借鉴了一篇名为“用于原始波形的非常深度卷积网络”的论文。² 您会发现它使用了我们在第三章中的许多构建模块，但是我们使用的是 1D 变体，而不是 2D 层，因为我们的音频输入少了一个维度：

```py
class AudioNet(nn.Module):
    def __init__(self):
        super(AudioNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 80, 4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(30)
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.avgPool(x)
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim = 2)
```

我们还需要一个优化器和一个损失函数。对于优化器，我们像以前一样使用 Adam，但你认为我们应该使用什么损失函数？（如果你回答`CrossEntropyLoss`，给自己一个金星！）

```py
audio_net = AudioNet()
audio_net.to(device)
```

创建完我们的模型后，我们保存我们的权重，并使用第四章中的`find_lr()`函数：

```py
audio_net.save("audionet.pth")
import torch.optim as optim
optimizer = optim.Adam(audionet.parameters(), lr=0.001)
logs,losses = find_lr(audio_net, nn.CrossEntropyLoss(), optimizer)
plt.plot(logs,losses)
```

从图 6-3 中的图表中，我们确定适当的学习率大约是`1e-5`（基于下降最陡的地方）。我们将其设置为我们的学习率，并重新加载我们模型的初始权重：

![AudioNet 学习率图](img/ppdl_0603.png)

###### 图 6-3\. AudioNet 学习率图

```py
lr = 1e-5
model.load("audionet.pth")
import torch.optim as optim
optimizer = optim.Adam(audionet.parameters(), lr=lr)
```

我们对模型进行 20 个周期的训练：

```py
train(audio_net, optimizer, torch.nn.CrossEntropyLoss(),
train_data_loader, valid_data_loader, epochs=20)
```

训练后，您应该发现模型在我们的数据集上达到了大约 13%至 17%的准确率。这比我们随机选择 50 个类别中的一个时可以期望的 2%要好。但也许我们可以做得更好；让我们探讨一种不同的查看音频数据的方式，可能会产生更好的结果。

# 这个频率是我的宇宙

如果您回顾一下 ESC-50 的 GitHub 页面，您会看到一个网络架构和其准确度得分的排行榜。您会注意到，与其他相比，我们的表现并不出色。我们可以扩展我们创建的模型使其更深，这可能会稍微提高我们的准确度，但要实现真正的性能提升，我们需要切换领域。在音频处理中，您可以像我们一直在做的那样处理纯波形；但大多数情况下，您将在*频域*中工作。这种不同的表示将原始波形转换为一个视图，显示了在特定时间点的所有声音频率。这可能是向神经网络呈现更丰富信息的表示形式，因为它可以直接处理这些频率，而不必弄清楚如何将原始波形信号映射为模型可以使用的内容。

让我们看看如何使用*LibROSA*生成频谱图。

## Mel 频谱图

传统上，进入频域需要在音频信号上应用傅立叶变换。我们将通过在 mel 刻度上生成我们的频谱图来超越这一点。*mel 刻度*定义了一个音高刻度，其中相距相等，其中 1000 mels = 1000 Hz。这种刻度在音频处理中很常用，特别是在语音识别和分类应用中。使用*LibROSA*生成 mel 频谱图只需要两行代码：

```py
sample_data, sr = librosa.load("ESC-50/train/1-100032-A-0.wav", sr=None)
spectrogram = librosa.feature.melspectrogram(sample_data, sr=sr)
```

这将生成一个包含频谱图数据的 NumPy 数组。如果我们像图 6-4 中所示显示这个频谱图，我们就可以看到我们声音中的频率：

```py
librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='mel')
```

![Mel 频谱图](img/ppdl_0604.png)

###### 图 6-4。Mel 频谱图

然而，图像中并没有太多信息。我们可以做得更好！如果我们将频谱图转换为对数刻度，由于该刻度能够表示更广泛的值范围，我们可以看到音频结构的更多内容。这在音频处理中很常见，*LibROSA*包含了一个方法：

```py
log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
```

这将计算一个`10 * log10(spectrogram / ref)`的缩放因子。`ref`默认为`1.0`，但在这里我们传入`np.max()`，以便`spectrogram / ref`将落在`[0,1]`的范围内。图 6-5 显示了新的频谱图。

![对数 Mel 频谱图](img/ppdl_0605.png)

###### 图 6-5。对数 mel 频谱图

现在我们有了一个对数刻度的 mel 频谱图！如果您调用`log_spectrogram.shape`，您会看到它是一个 2D 张量，这是有道理的，因为我们已经用张量绘制了图像。我们可以创建一个新的神经网络架构，并将这些新数据输入其中，但我有一个恶毒的技巧。我们刚刚生成了频谱图数据的图像。为什么不直接处理这些呢？

这一开始可能看起来有些愚蠢；毕竟，我们有基础频谱图数据，这比图像表示更精确（对我们来说，知道一个数据点是 58 而不是 60 对我们来说意义更大，而不是不同色调，比如紫色）。如果我们从头开始，这肯定是这样。但是！我们已经有了一些训练有素的网络，如 ResNet 和 Inception，我们*知道*它们擅长识别图像的结构和其他部分。我们可以构建音频的图像表示，并使用预训练网络再次利用迁移学习的超能力，通过很少的训练来大幅提高准确度。这对我们的数据集可能很有用，因为我们没有很多示例（只有 2000 个！）来训练我们的网络。

这个技巧可以应用于许多不同的数据集。如果您能找到一种便宜地将数据转换为图像表示的方法，那么值得这样做，并将 ResNet 网络应用于其，以了解迁移学习对您的作用，这样您就知道通过使用不同方法可以超越什么。有了这个，让我们创建一个新的数据集，以便根据需要为我们生成这些图像。

## 一个新的数据集

现在丢弃原始的`ESC50`数据集类，构建一个新的`ESC50Spectrogram`。虽然这将与旧类共享一些代码，但在这个版本的`__get_item__`方法中会有更多的操作。我们通过*LibROSA*生成频谱图，然后通过一些复杂的`matplotlib`操作将数据转换为 NumPy 数组。我们将该数组应用于我们的转换流水线（只使用`ToTensor`），并返回该数组和项目的标签。以下是代码：

```py
class ESC50Spectrogram(Dataset):

def __init__(self,path):
    files = Path(path).glob('*.wav')
    self.items = [(f,int(f.name.split("-")[-1].replace(".wav","")))
                   for f in files]
    self.length = len(self.items)
    self.transforms = torchvision.transforms.Compose(
                 [torchvision.transforms.ToTensor()])

def __getitem__(self, index):
    filename, label = self.items[index]
    audio_tensor, sample_rate = librosa.load(filename, sr=None)
    spectrogram = librosa.feature.melspectrogram(audio_tensor, sr=sample_rate)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sample_rate,
                             x_axis='time', y_axis='mel')
    plt.gcf().canvas.draw()
    audio_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    audio_data = audio_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return (self.transforms(audio_data), label)

def __len__(self):
    return self.length
```

我们不会花太多时间在这个数据集的版本上，因为它有一个很大的缺陷，我用 Python 的`process_time()`方法演示了这一点：

```py
oldESC50 = ESC50("ESC-50/train/")
start_time = time.process_time()
oldESC50.__getitem__(33)
end_time = time.process_time()
old_time = end_time - start_time

newESC50 = ESC50Spectrogram("ESC-50/train/")
start_time = time.process_time()
newESC50.__getitem__(33)
end_time = time.process_time()
new_time = end_time - start_time

old_time = 0.004786839000075815
new_time = 0.39544327499993415
```

新数据集的速度几乎比我们原始的只返回原始音频的数据集慢一百倍！这将使训练变得非常缓慢，甚至可能抵消使用迁移学习所能带来的任何好处。

我们可以使用一些技巧来解决大部分问题。第一种方法是添加一个缓存，将生成的频谱图存储在内存中，这样我们就不必每次调用`__getitem__`方法时都重新生成它。使用 Python 的`functools`包，我们可以很容易地做到这一点：

```py
import functools

class ESC50Spectrogram(Dataset):
 #skipping init code

    @functools.lru_cache(maxsize=<size of dataset>)
    def __getitem__(self, index):
```

只要您有足够的内存来存储整个数据集的内容到 RAM 中，这可能就足够了。我们设置了一个*最近最少使用*（LRU）缓存，将尽可能长时间地保留内容在内存中，最近没有被访问的索引在内存紧张时首先被驱逐出缓存。然而，如果您没有足够的内存来存储所有内容，您将在每个批次迭代时遇到减速，因为被驱逐的频谱图需要重新生成。

我的首选方法是*预计算*所有可能的图表，然后创建一个新的自定义数据集类，从磁盘加载这些图像。（您甚至可以添加 LRU 缓存注释以进一步加快速度。）

我们不需要为预计算做任何花哨的事情，只需要一个将图表保存到正在遍历的目录中的方法：

```py
def precompute_spectrograms(path, dpi=50):
    files = Path(path).glob('*.wav')
    for filename in files:
        audio_tensor, sample_rate = librosa.load(filename, sr=None)
        spectrogram = librosa.feature.melspectrogram(audio_tensor, sr=sr)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        librosa.display.specshow(log_spectrogram, sr=sr, x_axis='time',
                                 y_axis='mel')
        plt.gcf().savefig("{}{}_{}.png".format(filename.parent,dpi,
                          filename.name),dpi=dpi)
```

这种方法比我们之前的数据集更简单，因为我们可以使用`matplotlib`的`savefig`方法直接将图表保存到磁盘，而不必与 NumPy 搞在一起。我们还提供了一个额外的输入参数`dpi`，允许我们控制生成输出的质量。在我们已经设置好的所有`train`、`test`和`valid`路径上运行（可能需要几个小时才能处理完所有图像）。

现在我们只需要一个新的数据集来读取这些图像。我们不能使用第二章到第四章中的标准`ImageDataLoader`，因为 PNG 文件名方案与其使用的目录结构不匹配。但没关系，我们可以使用 Python Imaging Library 打开一张图片：

```py
from PIL import Image

    class PrecomputedESC50(Dataset):
        def __init__(self,path,dpi=50, transforms=None):
            files = Path(path).glob('{}*.wav.png'.format(dpi))
            self.items = [(f,int(f.name.split("-")[-1]
            .replace(".wav.png",""))) for f in files]
            self.length = len(self.items)
            if transforms=None:
                self.transforms =
                torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            else:
                self.transforms = transforms

        def __getitem__(self, index):
            filename, label = self.items[index]
            img = Image.open(filename)
            return (self.transforms(img), label)

        def __len__(self):
            return self.length
```

这段代码更简单，希望从数据集中获取一个条目所需的时间也反映在其中：

```py
start_time = time.process_time()
b.__getitem__(33)
end_time = time.process_time()
end_time - start_time
>> 0.0031465259999094997
```

从这个数据集获取一个元素大致需要与我们原始基于音频的数据集相同的时间，所以我们不会因为转向基于图像的方法而失去任何东西，除了预计算所有图像并创建数据库的一次性成本。我们还提供了一个默认的转换流水线，将图像转换为张量，但在初始化期间可以替换为不同的流水线。有了这些优化，我们可以开始将迁移学习应用到这个问题上。

## 一只野生的 ResNet 出现了

正如您可能记得的那样，从第四章中，迁移学习要求我们使用已经在特定数据集上训练过的模型（在图像的情况下，可能是 ImageNet），然后在我们特定的数据领域上微调它，即我们将 ESC-50 数据集转换为频谱图像。您可能会想知道一个在*正常*照片上训练过的模型对我们是否有用。事实证明，预训练模型*确实*学到了很多结构，可以应用于乍看起来可能非常不同的领域。以下是我们从第四章中初始化模型的代码：

```py
from torchvision import models
spec_resnet = models.ResNet50(pretrained=True)

for param in spec_resnet.parameters():
    param.requires_grad = False

spec_resnet.fc = nn.Sequential(nn.Linear(spec_resnet.fc.in_features,500),
nn.ReLU(),
nn.Dropout(), nn.Linear(500,50))
```

这使我们使用了一个预训练的（并冻结的）`ResNet50`模型，并将模型的头部替换为一个未经训练的`Sequential`模块，最后以一个输出为 50 的`Linear`结尾，每个类别对应 ESC-50 数据集中的一个类。我们还需要创建一个`DataLoader`，以获取我们预先计算的频谱图。当我们创建 ESC-50 数据集时，我们还希望使用标准的 ImageNet 标准差和均值对传入的图像进行归一化，因为预训练的 ResNet-50 架构就是用这种方式训练的。我们可以通过传入一个新的管道来实现：

```py
esc50pre_train = PreparedESC50(PATH, transforms=torchvision.transforms
.Compose([torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize
(mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225])]))

esc50pre_valid = PreparedESC50(PATH, transforms=torchvision.transforms
.Compose([torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize
(mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225])]))

esc50_train_loader = (esc50pre_train, bs, shuffle=True)
esc50_valid_loader = (esc50pre_valid, bs, shuffle=True)
```

设置好数据加载器后，我们可以继续寻找学习率并准备训练。

## 寻找学习率

我们需要找到一个学习率来在我们的模型中使用。就像在第四章中一样，我们会保存模型的初始参数，并使用我们的`find_lr()`函数来找到一个适合训练的学习率。图 6-6 显示了损失与学习率之间的关系图。

```py
spec_resnet.save("spec_resnet.pth")
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(spec_resnet.parameters(), lr=lr)
logs,losses = find_lr(spec_resnet, loss_fn, optimizer)
plt.plot(logs, losses)
```

![SpecResNet 学习率图](img/ppdl_0606.png)

###### 图 6-6\. SpecResNet 学习率图

查看学习率与损失之间的图表，似乎`1e-2`是一个不错的起点。由于我们的 ResNet-50 模型比之前的模型更深，我们还将使用不同的学习率`[1e-2,1e-4,1e-8]`，其中最高的学习率应用于我们的分类器（因为它需要最多的训练！），而对于已经训练好的主干使用较慢的学习率。同样，我们使用 Adam 作为优化器，但可以尝试其他可用的优化器。

在我们应用这些不同的学习率之前，我们会训练几个周期，只更新分类器，因为我们在创建网络时*冻结*了 ResNet-50 的主干：

```py
optimizer = optim.Adam(spec_resnet.parameters(), lr=[1e-2,1e-4,1e-8])

train(spec_resnet, optimizer, nn.CrossEntropyLoss(),
esc50_train_loader, esc50_val_loader,epochs=5,device="cuda")
```

现在我们解冻主干并应用我们的不同学习率：

```py
for param in spec_resnet.parameters():
    param.requires_grad = True

optimizer = optim.Adam(spec_resnet.parameters(), lr=[1e-2,1e-4,1e-8])

train(spec_resnet, optimizer, nn.CrossEntropyLoss(),
esc50_train_loader, esc50_val_loader,epochs=20,device="cuda")

> Epoch 19, accuracy = 0.80
```

如您所见，验证准确率约为 80%，我们已经远远超过了原始的`AudioNet`模型。再次展示了迁移学习的强大！可以继续训练更多周期，看看准确率是否继续提高。如果查看 ESC-50 排行榜，我们已经接近人类水平的准确率。而这仅仅是使用 ResNet-50。您可以尝试使用 ResNet-101，或者尝试使用不同架构的集成来进一步提高分数。

还有数据增强要考虑。让我们看看在迄今为止我们一直在工作的两个领域中如何做到这一点。

# 音频数据增强

当我们在第四章中查看图像时，我们发现通过对传入的图片进行更改，如翻转、裁剪或应用其他转换，可以提高分类器的准确性。通过这些方式，我们让神经网络在训练阶段更加努力，并在最后得到一个更*通用*的模型，而不仅仅是适应所呈现的数据（过拟合的祸根，不要忘记）。我们能在这里做同样的事情吗？是的！事实上，我们可以使用两种方法——一种明显的方法适用于原始音频波形，另一种可能不太明显的想法源自我们决定在 mel 频谱图像上使用基于 ResNet 的分类器。让我们先看看音频转换。

## torchaudio 转换

与`torchvision`类似，`torchaudio`包括一个`transforms`模块，对传入数据执行转换。然而，提供的转换数量有些稀少，特别是与处理图像时得到的丰富多样相比。如果您感兴趣，请查看[文档](https://oreil.ly/d1kp6)获取完整列表，但我们在这里只看一个`torchaudio.transforms.PadTrim`。在 ESC-50 数据集中，每个音频剪辑的长度都是相同的。这在现实世界中并不常见，但我们的神经网络喜欢（有时也坚持，取决于它们的构建方式）输入数据是规则的。`PadTrim`将接收到的音频张量填充到所需长度，或者将其修剪到不超过该长度。如果我们想将剪辑修剪到新长度，我们会这样使用`PadTrim`：

```py
audio_tensor, rate = torchaudio.load("test.wav")
audio_tensor.shape
trimmed_tensor = torchaudio.transforms.PadTrim(max_len=1000)(audio_orig)
```

然而，如果您正在寻找实际改变音频声音的增强（例如添加回声、噪音或更改剪辑的节奏），那么`torchaudio.transforms`模块对您没有用。相反，我们需要使用*SoX*。

## SoX 效果链

为什么它不是`transforms`模块的一部分，我真的不确定，但`torchaudio.sox_effects.SoxEffectsChain`允许您创建一个或多个*SoX*效果链，并将这些效果应用于输入文件。界面有点棘手，让我们在一个新版本的数据集中看看它的运行方式，该数据集改变了音频文件的音调：

```py
class ESC50WithPitchChange(Dataset):

    def __init__(self,path):
        # Get directory listing from path
        files = Path(path).glob('*.wav')
        # Iterate through the listing and create a list of tuples (filename, label)
        self.items = [(f,f.name.split("-")[-1].replace(".wav","")) for f in files]
        self.length = len(self.items)
        self.E = torchaudio.sox_effects.SoxEffectsChain()
        self.E.append_effect_to_chain("pitch", [0.5])

    def __getitem__(self, index):
        filename, label = self.items[index]
        self.E.set_input_file(filename)
        audio_tensor, sample_rate = self.E.sox_build_flow_effects()
        return audio_tensor, label

    def __len__(self):
        return self.length
```

在我们的`__init__`方法中，我们创建一个新的实例变量`E`，一个`SoxEffectsChain`，它将包含我们要应用于音频数据的所有效果。然后，我们通过使用`append_effect_to_chain`添加一个新效果，该方法接受一个指示效果名称的字符串，以及要发送给`sox`的参数数组。您可以通过调用`torchaudio.sox_effects.effect_names()`获取可用效果的列表。如果我们要添加另一个效果，它将在我们已经设置的音调效果之后发生，因此如果您想创建一系列单独的效果并随机应用它们，您需要为每个效果创建单独的链。

当选择要返回给数据加载器的项目时，情况有所不同。我们不再使用`torchaudio.load()`，而是引用我们的效果链，并使用`set_input_file`指向文件。但请注意，这并不会加载文件！相反，我们必须使用`sox_build_flow_effects()`，它在后台启动*SoX*，应用链中的效果，并返回我们通常从`load()`中获得的张量和采样率信息。

*SoX*可以做的事情数量相当惊人，我不会详细介绍您可以使用的所有可能效果。我建议结合`list_effects()`查看[*SoX*文档](https://oreil.ly/uLBTF)以了解可能性。

这些转换允许我们改变原始音频，但在本章的大部分时间里，我们都在构建一个处理管道，用于处理梅尔频谱图像。我们可以像为该管道生成初始数据集所做的那样，创建修改后的音频样本，然后从中创建频谱图像，但在那时，我们将创建大量数据，需要在运行时混合在一起。幸运的是，我们可以对频谱图像本身进行一些转换。

## SpecAugment

现在，你可能会想到：“等等，这些频谱图只是图片！我们可以对它们使用任何图片变换！” 是的！后面的你获得金星。但是我们必须小心一点；例如，随机裁剪可能会剪掉足够的频率，从而潜在地改变输出类别。在我们的 ESC-50 数据集中，这不是一个大问题，但如果你在做类似语音识别的事情，那么在应用增强时肯定要考虑这一点。另一个有趣的可能性是，因为我们知道所有的频谱图具有相同的结构（它们总是一个频率图！），我们可以创建基于图像的变换，专门围绕这种结构工作。

2019 年，谷歌发布了一篇关于 SpecAugment 的论文，[3]在许多音频数据集上报告了新的最先进结果。该团队通过使用三种新的数据增强技术直接应用于梅尔频谱图：时间弯曲、频率掩码和时间掩码，从中获得了这些结果。我们不会讨论时间弯曲，因为从中获得的好处很小，但我们将为掩码时间和频率实现自定义变换。

### 频率掩码

*频率掩码*会随机地从我们的音频输入中移除一个频率或一组频率。这样做是为了让模型更加努力；它不能简单地*记忆*输入及其类别，因为在每个批次中输入的不同频率会被掩码。模型将不得不学习其他特征，以确定如何将输入映射到一个类别，这希望会导致一个更准确的模型。

在我们的梅尔频谱图中，这通过确保在任何时间步长中该频率的频谱图中没有任何内容来显示。图 6-7 展示了这是什么样子：基本上是在自然频谱图上画了一条空白线。

这是一个实现频率掩码的自定义`Transform`的代码：

```py
class FrequencyMask(object):
    """
 Example:
 >>> transforms.Compose([
 >>>     transforms.ToTensor(),
 >>>     FrequencyMask(max_width=10, use_mean=False),
 >>> ])

 """

    def __init__(self, max_width, use_mean=True):
        self.max_width = max_width
        self.use_mean = use_mean

    def __call__(self, tensor):
        """
 Args:
 tensor (Tensor): Tensor image of
 size (C, H, W) where the frequency
 mask is to be applied.

 Returns:
 Tensor: Transformed image with Frequency Mask.
 """
        start = random.randrange(0, tensor.shape[2])
        end = start + random.randrange(1, self.max_width)
        if self.use_mean:
            tensor[:, start:end, :] = tensor.mean()
        else:
            tensor[:, start:end, :] = 0
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + "(max_width="
        format_string += str(self.max_width) + ")"
        format_string += 'use_mean=' + (str(self.use_mean) + ')')

        return format_string
```

当应用变换时，PyTorch 将使用图像的张量表示调用`__call__`方法（因此我们需要将其放在将图像转换为张量后的`Compose`链中，而不是之前）。我们假设张量将以*通道×高度×宽度*格式呈现，并且我们希望将高度值设置在一个小范围内，要么为零，要么为图像的平均值（因为我们使用对数梅尔频谱图，平均值应该与零相同，但我们包括两种选项，以便您可以尝试看哪种效果更好）。范围由`max_width`参数提供，我们得到的像素掩码将在 1 到`max_pixels`之间。我们还需要为掩码选择一个随机起点，这就是`start`变量的作用。最后，这个变换的复杂部分——我们应用我们生成的掩码：

```py
tensor[:, start:end, :] = tensor.mean()
```

当我们将其分解时，情况就没有那么糟糕了。我们的张量有三个维度，但我们希望在所有红色、绿色和蓝色通道上应用这个变换，所以我们使用裸的`:`来选择该维度中的所有内容。使用`start:end`，我们选择我们的高度范围，然后我们选择宽度通道中的所有内容，因为我们希望在每个时间步长上应用我们的掩码。然后在表达式的右侧，我们设置值；在这种情况下，是`tensor.mean()`。如果我们从 ESC-50 数据集中取一个随机张量并将变换应用于它，我们可以在图 6-7 中看到这个类别正在创建所需的掩码。

```py
torchvision.transforms.Compose([FrequencyMask(max_width=10, use_mean=False),
torchvision.transforms.ToPILImage()])(torch.rand(3,250,200))
```

![应用于随机 ESC-50 样本的频率掩码](img/ppdl_0607.png)

###### 图 6-7。应用于随机 ESC-50 样本的频率掩码

接下来我们将转向时间掩码。

### 时间掩码

有了我们的频率掩码完成后，我们可以转向*时间掩码*，它与频率掩码相同，但在时间域中。这里的代码大部分是相同的：

```py
class TimeMask(object):
    """
 Example:
 >>> transforms.Compose([
 >>>     transforms.ToTensor(),
 >>>     TimeMask(max_width=10, use_mean=False),
 >>> ])

 """

    def __init__(self, max_width, use_mean=True):
        self.max_width = max_width
        self.use_mean = use_mean

    def __call__(self, tensor):
        """
 Args:
 tensor (Tensor): Tensor image of
 size (C, H, W) where the time mask
 is to be applied.

 Returns:
 Tensor: Transformed image with Time Mask.
 """
        start = random.randrange(0, tensor.shape[1])
        end = start + random.randrange(0, self.max_width)
        if self.use_mean:
            tensor[:, :, start:end] = tensor.mean()
        else:
            tensor[:, :, start:end] = 0
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + "(max_width="
        format_string += str(self.max_width) + ")"
        format_string += 'use_mean=' + (str(self.use_mean) + ')')
        return format_string
```

正如您所看到的，这个类与频率掩码类似。唯一的区别是我们的`start`变量现在在高度轴上的某个点范围内，当我们进行掩码处理时，我们这样做：

```py
tensor[:, :, start:end] = 0
```

这表明我们选择张量的前两个维度的所有值和最后一个维度中的`start:end`范围。再次，我们可以将这应用于来自 ESC-50 的随机张量，以查看掩码是否被正确应用，如图 6-8 所示。

```py
torchvision.transforms.Compose([TimeMask(max_width=10, use_mean=False),
torchvision.transforms.ToPILImage()])(torch.rand(3,250,200))
```

![应用于随机 ESC-50 样本的时间掩码](img/ppdl_0608.png)

###### 图 6-8。应用于随机 ESC-50 样本的时间掩码

为了完成我们的增强，我们创建一个新的包装器转换，确保一个或两个掩码应用于频谱图像：

```py
class PrecomputedTransformESC50(Dataset):
    def __init__(self,path,dpi=50):
        files = Path(path).glob('{}*.wav.png'.format(dpi))
        self.items = [(f,f.name.split("-")[-1].replace(".wav.png",""))
                      for f in files]
        self.length = len(self.items)
        self.transforms = transforms.Compose([
    transforms.ToTensor(),
    RandomApply([FrequencyMask(self.max_freqmask_width)]p=0.5),
    RandomApply([TimeMask(self.max_timemask_width)]p=0.5)
])

    def __getitem__(self, index):
        filename, label = self.items[index]
        img = Image.open(filename)
        return (self.transforms(img), label)

    def __len__(self):
        return self.length
```

尝试使用这种数据增强重新运行训练循环，看看您是否像谷歌一样通过这些掩码获得更好的准确性。但也许我们还可以尝试更多与这个数据集有关的内容？

# 进一步实验

到目前为止，我们已经创建了两个神经网络——一个基于原始音频波形，另一个基于 mel 频谱图像——用于对 ESC-50 数据集中的声音进行分类。尽管您已经看到基于 ResNet 的模型在使用迁移学习的力量时更准确，但创建这两个网络的组合来查看是否增加或减少准确性将是一个有趣的实验。这样做的一个简单方法是重新审视第四章中的集成方法：只需组合和平均预测。此外，我们跳过了基于我们从频谱图中获取的原始数据构建网络的想法。如果创建了一个适用于该数据的模型，那么如果将其引入集成，是否会提高整体准确性？我们还可以使用其他版本的 ResNet，或者我们可以创建使用不同预训练模型（如 VGG 或 Inception）作为骨干的新架构。探索一些这些选项并看看会发生什么；在我的实验中，SpecAugment 将 ESC-50 分类准确性提高了约 2%。

# 结论

在本章中，我们使用了两种非常不同的音频分类策略，简要介绍了 PyTorch 的`torchaudio`库，并看到了在数据集上预先计算转换的方法，而在进行实时转换时会严重影响训练时间。我们讨论了两种数据增强方法。作为一个意外的奖励，我们再次通过使用迁移学习来训练基于图像的模型，快速生成一个与 ESC-50 排行榜上其他模型相比准确度较高的分类器。

这结束了我们对图像、测试和音频的导览，尽管我们将在第九章中再次涉及这三个方面，当我们看一些使用 PyTorch 的应用程序时。不过，接下来，我们将看看在模型训练不够正确或速度不够快时如何调试模型。

# 进一步阅读

+   [“解释和说明用于音频信号分类的深度神经网络”](https://arxiv.org/abs/1807.03418) 由 Sören Becker 等人（2018 年）撰写

+   [“用于大规模音频分类的 CNN 架构”](https://arxiv.org/abs/1609.09430v2) 由 Shawn Hershey 等人（2016 年）

理解*SoX*可以做什么超出了本书的范围，并且对于我们接下来在本章中要做的事情并不是必要的。

参见[“用于原始波形的非常深度卷积神经网络”](https://arxiv.org/pdf/1610.00087.pdf) 由 Wei Dai 等人（2016 年）。

参见[“SpecAugment：用于自动语音识别的简单数据增强方法”](https://arxiv.org/abs/1904.08779) 由 Daniel S. Park 等人（2019 年）。
