# 第十五章。音乐和深度学习

这本书中的其他章节都是关于图像或文本的处理。这些章节代表了深度学习研究中媒体的平衡，但这并不意味着声音处理不是有趣的，我们在过去几年中也看到了一些重大进展。语音识别和语音合成使得像亚马逊 Alexa 和谷歌 Home 这样的家庭助手成为可能。自从 Siri 推出以来，那个老的情景喜剧笑话中电话拨错号码的情节并不是很现实。

开始尝试这些系统很容易；有一些 API 可以让你在几个小时内建立一个简单的语音应用程序。然而，语音处理是在亚马逊、谷歌或苹果的数据中心进行的，所以我们不能真的将这些视为深度学习实验。构建最先进的语音识别系统很困难，尽管 Mozilla 的 Deep Speech 正在取得一些令人印象深刻的进展。

这一章重点是音乐。我们将首先训练一个音乐分类模型，可以告诉我们正在听什么音乐。然后我们将使用这个模型的结果来索引本地 MP3，使得可以找到风格相似的歌曲。之后我们将使用 Spotify API 创建一个公共播放列表的语料库，用来训练音乐推荐系统。

本章的笔记本有：

```py
15.1 Song Classification
15.2 Index Local MP3s
15.3 Spotify Playlists
15.4 Train a Music Recommender
```

# 15.1 为音乐分类创建训练集

## 问题

如何获取和准备一组音乐用于分类？

## 解决方案

从加拿大维多利亚大学提供的测试集中创建频谱图。

你可以尝试通过连接那个带有 MP3 收藏的尘封外部驱动器，并依赖那些歌曲的标签来做这件事。但很多标签可能有些随机或缺失，所以最好从一个科学机构获得一个有标签的训练集开始：

```py
wget http://opihi.cs.uvic.ca/sound/genres.tar.gz
tar xzf genres.tar.gz
```

这将为我们创建一个包含不同流派音乐的子目录*genres*：

```py
>ls ~/genres
blues  classical  country  disco  hiphop  jazz  metal  pop  reggae  rock
```

这些目录包含声音文件（*.au*），每种流派 100 个片段，每个片段长 29 秒。我们可以尝试直接将原始声音帧馈送到网络中，也许 LSTM 会捕捉到一些东西，但有更好的声音预处理方法。声音实际上是声波，但我们听不到波。相反，我们听到一定频率的音调。

因此，让我们的网络更像我们的听觉工作的一个好方法是将声音转换为频谱图块；每个样本将由一系列音频频率及其相应的强度表示。Python 的`librosa`库有一些标准函数可以做到这一点，并且还提供了所谓的*melspectrogram*，一种旨在紧密模拟人类听觉工作方式的频谱图。所以让我们加载音乐并将片段转换为 melspectrograms：

```py
def load_songs(song_folder):
    song_specs = []
    idx_to_genre = []
    genre_to_idx = {}
    genres = []
    for genre in os.listdir(song_folder):
        genre_to_idx[genre] = len(genre_to_idx)
        idx_to_genre.append(genre)
        genre_folder = os.path.join(song_folder, genre)
        for song in os.listdir(genre_folder):
            if song.endswith('.au'):
                signal, sr = librosa.load(
                    os.path.join(genre_folder, song))
                melspec = librosa.feature.melspectrogram(
                    signal, sr=sr).T[:1280,]
                song_specs.append(melspec)
                genres.append(genre_to_idx[genre])
    return song_specs, genres, genre_to_idx, idx_to_genre
```

让我们也快速看一下一些流派的频谱图。由于这些频谱图现在只是矩阵，我们可以将它们视为位图。它们实际上非常稀疏，所以我们将过度曝光它们以查看更多细节：

```py
def show_spectogram(show_genre):
    show_genre = genre_to_idx[show_genre]
    specs = []
    for spec, genre in zip(song_specs, genres):
        if show_genre == genre:
            specs.append(spec)
            if len(specs) == 25:
                break
    if not specs:
        return 'not found!'
    x = np.concatenate(specs, axis=1)
    x = (x - x.min()) / (x.max() - x.min())
    plt.imshow((x *20).clip(0, 1.0))

show_spectogram('classical')
```

![古典音乐频谱图](img/dlcb_15in01.png)

```py
show_spectogram('metal')
```

![金属音乐频谱图](img/dlcb_15in02.png)

尽管很难说图片到底代表什么意思，但有一些迹象表明金属音乐可能比古典音乐更具有刚性结构，这也许并不完全出乎意料。

## 讨论

正如我们在整本书中看到的那样，在让网络处理数据之前对数据进行预处理会显著增加我们成功的机会。在声音处理方面，`librosa`有几乎任何你想要的功能，从加载声音文件并在笔记本中播放它们到可视化它们和进行任何类型的预处理。

通过视觉检查频谱图并不能告诉我们太多，但它确实给了我们一个暗示，即不同音乐流派的频谱图是不同的。在下一个步骤中，我们将看到网络是否也能学会区分它们。

# 15.2 训练音乐流派检测器

## 问题

如何设置和训练一个深度网络来检测音乐流派？

## 解决方案

使用一维卷积网络。

在本书中，我们已经使用卷积网络进行图像检测（参见第九章）和文本（参见第七章）。处理我们的频谱图像为图像可能是更合乎逻辑的方法，但实际上我们将使用一维卷积网络。我们的频谱图中的每个帧代表音乐的一个帧。当我们尝试对流派进行分类时，使用卷积网络将时间段转换为更抽象的表示是有意义的；减少帧的“高度”在直觉上不太合理。

我们将从顶部堆叠一些层。这将把我们的输入从 128 维减少到 25。`GlobalMaxPooling`层将把这个转换成一个 128 浮点向量：

```py
inputs = Input(input_shape)
x = inputs
for layers in range(3):
x = Conv1D(128, 3, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=6, strides=2)(x)
x = GlobalMaxPooling1D()(x)
```

接着是一些全连接层，以获得标签：

```py
for fc in range(2):
x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(10, activation='softmax')(x)
```

在将数据输入模型之前，我们将每首歌曲分成 10 个每个 3 秒的片段。我们这样做是为了增加数据量，因为 1000 首歌曲并不算太多：

```py
def split_10(x, y):
    s = x.shape
    s = (s[0] * 10, s[1] // 10, s[2])
    return x.reshape(s), np.repeat(y, 10, axis=0)

genres_one_hot = keras.utils.to_categorical(
    genres, num_classes=len(genre_to_idx))

x_train, x_test, y_train, y_test = train_test_split(
    np.array(song_specs), np.array(genres_one_hot),
    test_size=0.1, stratify=genres)

x_test, y_test = split_10(x_test, y_test)
x_train, y_train = split_10(x_train, y_train)
```

在 100 个 epochs 后，训练这个模型可以达到大约 60%的准确率，这并不差，但肯定不是超人的水平。我们可以通过利用将每首歌曲分成 10 个片段并使用跨片段的信息来改进结果。多数投票可能是一种策略，但事实证明，选择模型最确定的片段效果更好。我们可以通过将数据重新分成 100 个片段，并在每个片段上应用`argmax`来实现这一点。这将为每个片段得到整个片段中的索引。通过应用模 10，我们可以得到我们标签集中的索引：

```py
def unsplit(values):
    chunks = np.split(values, 100)
    return np.array([np.argmax(chunk) % 10 for chunk in chunks])

predictions = unsplit(model.predict(x_test))
truth = unsplit(y_test)
accuracy_score(predictions, truth)
```

这让我们达到了 75%的准确率。

## 讨论

对于我们的 10 种流派，每种有 100 首歌曲，我们没有太多的训练数据。将我们的歌曲分成 10 个每个 3 秒的块，可以让我们达到一定程度的准确性，尽管我们的模型仍然有点过拟合。

探索的一个方向是应用一些数据增强技术。我们可以尝试给音乐添加噪音，稍微加快或减慢速度，尽管频谱图本身可能并不会有太大变化。最好是能获取更大量的音乐数据。

# 15.3 可视化混淆

## 问题

如何清晰地展示网络所犯的错误？

## 解决方案

以图形方式显示混淆矩阵。

混淆矩阵的列代表真实的流派，行代表模型预测的流派。单元格包含每个（真实，预测）对的计数。`sklearn`带有一个方便的方法来计算它：

```py
cm = confusion_matrix(pred_values, np.argmax(y_test, axis=1))
print(cm)
```

```py
[[65 13  0  6  5  1  4  5  2  1]
 [13 54  1  3  4  0 20  1  0  9]
 [ 5  2 99  0  0  0 12 33  0  2]
 [ 0  0  0 74 29  1  8  0 18 10]
 [ 0  0  0  2 55  0  0  1  2  0]
 [ 1  0  0  1  0 95  0  0  0  6]
 [ 8 17  0  2  5  2 45  0  1  4]
 [ 4  4  0  1  2  0 10 60  1  4]
 [ 0  1  0  1  0  1  0  0 64  5]
 [ 4  9  0 10  0  0  1  0 12 59]]
```

我们可以通过给矩阵着色来更清晰地可视化。将矩阵转置，这样我们可以看到每行的混淆，也让处理变得更容易：

```py
plt.imshow(cm.T, interpolation='nearest', cmap='gray')
plt.xticks(np.arange(0, len(idx_to_genre)), idx_to_genre)
plt.yticks(np.arange(0, len(idx_to_genre)), idx_to_genre)

plt.show()
```

![混淆矩阵](img/dlcb_15in03.png)

## 讨论

混淆矩阵是展示网络性能的一种巧妙方式，但它也让你了解它出错的地方，这可能暗示着如何改进事情。在这个示例中，我们可以看到网络在区分古典音乐和金属音乐与其他类型音乐时表现得非常好，但在区分摇滚和乡村音乐时表现得不太好。当然，这一切都是意料之中的。

# 15.4 索引现有音乐

## 问题

您想建立一个能捕捉音乐风格的音乐片段索引。

## 解决方案

将模型的最后一个全连接层视为嵌入层。

在第十章中，我们通过将图像识别网络的最后一个全连接层解释为图像嵌入，构建了一个图像的反向搜索引擎。我们可以用音乐做类似的事情。让我们开始收集一些 MP3 文件——你可能有一些散落在某处的收藏：

```py
MUSIC_ROOT = _</path/to/music>_
mp3s = []
for root, subdirs, files in os.walk(MUSIC_ROOT):
    for fn in files:
        if fn.endswith('.mp3'):
            mp3s.append(os.path.join(root, fn))
```

然后我们将对它们进行索引。与以前一样，我们提取一个梅尔频谱图。我们还获取 MP3 标签：

```py
def process_mp3(path):
    tag = TinyTag.get(path)
    signal, sr = librosa.load(path,
                              res_type='kaiser_fast',
                              offset=30,
                              duration=30)
    melspec = librosa.feature.melspectrogram(signal, sr=sr).T[:1280,]
        if len(melspec) != 1280:
            return None
    return {'path': path,
            'melspecs': np.asarray(np.split(melspec, 10)),
            'tag': tag}

songs = [process_mp3(path) for path in tqdm(mp3s)]
songs = [song for song in songs if song]
```

我们想要索引所有 MP3 的每个频谱图像-如果我们将它们全部连接在一起，我们可以一次完成：

```py
inputs = []
for song in songs:
    inputs.extend(song['melspecs'])
inputs = np.array(inputs)
```

为了获得向量表示，我们将构建一个模型，该模型从我们先前模型的倒数第四层返回，并在收集的频谱上运行：

```py
cnn_model = load_model('zoo/15/song_classify.h5')
vectorize_model = Model(inputs=cnn_model.input,
                        outputs=cnn_model.layers[-4].output)
vectors = vectorize_model.predict(inputs)
```

一个简单的最近邻模型让我们现在可以找到相似的歌曲。给定一首歌曲，我们将查找它的每个向量的其他最近向量是什么。我们可以跳过第一个结果，因为它是向量本身：

```py
nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(vectors)
def most_similar_songs(song_idx):
    distances, indices = nbrs.kneighbors(
        vectors[song_idx * 10: song_idx * 10 + 10])
    c = Counter()
    for row in indices:
        for idx in row[1:]:
            c[idx // 10] += 1
    return c.most_common()
```

在随机歌曲上尝试这个似乎有效：

```py
song_idx = 7
print(songs[song_idx]['path'])

print('---')
for idx, score in most_similar_songs(song_idx)[:5]:
    print(songs[idx]['path'], score)
print('')
```

```py
00 shocking blue - Venus (yes the.mp3
---
00 shocking blue - Venus (yes the.mp3 20
The Shocking Blue/Have A Nice Day_ Vol 1/00 Venus.mp3 12
The Byrds/00 Eve of Destruction.mp3 12
Goldfinger _ Weezer _ NoFx _ L/00 AWESOME.mp3 6
```

使用我们模型的最后一个全连接层对歌曲进行索引工作得相当不错。在这个例子中，它不仅找到了原始歌曲，还找到了一个略有不同版本的歌曲，恰好在 MP3 收藏中。其他两首返回的歌曲是否在风格上真的相似是一个判断问题，但它们并不完全不同。

这里的代码可以用作构建类似 Shazam 的基础；录制一小段音乐，通过我们的向量化器运行，看看它与哪首索引歌曲最接近。Shazam 的算法是不同的，并且早于深度学习的流行。

通过找到听起来相似的其他音乐来获取音乐推荐系统的基础。然而，它只适用于我们已经可以访问的音乐，这在一定程度上限制了其实用性。在本章的其余部分，我们将看看另一种构建音乐推荐系统的方法。

# 15.5 设置 Spotify API 访问权限

## 问题

如何获得大量音乐数据的访问权限？

## 解决方案

使用 Spotify API。

我们在上一个配方中创建的系统是一种音乐推荐器，但它只推荐它已经看过的歌曲。通过从 Spotify API 中收集播放列表和歌曲，我们可以建立一个更大的训练集。让我们从 Spotify 注册一个新应用程序开始。转到[*https://beta.developer.spotify.com/dashboard/applications*](https://beta.developer.spotify.com/dashboard/applications)，创建一个新应用程序。

###### 注意

这里提到的 URL 以 beta 开头。当您阅读此内容时，Spotify 上的新应用程序界面可能已经退出 beta 测试，URL 可能已更改。

您需要先登录，可能需要注册。创建应用程序后，转到应用程序页面并记下客户端 ID 和客户端密钥。由于密钥是秘密的，您需要按下按钮显示它。

现在在三个常量中输入您的各种细节：

```py
CLIENT_ID = '<*`your client id`*>'
CLIENT_SECRET = '<*`your secret`*>'
USER_ID = '<*`your user id`*>'

```

您现在可以访问 Spotify API：

```py
uri = 'http://127.0.0.1:8000/callback'
token = util.prompt_for_user_token(USER_ID, '',
                                   client_id=CLIENT_ID,
                                   client_secret=CLIENT_SECRET,
                                   redirect_uri=uri)
session = spotipy.Spotify(auth=token)
```

第一次运行此代码时，API 将要求您在浏览器中输入一个 URL。当从笔记本运行时，这种方式有些笨拙；要重定向的 URL 将打印在笔记本服务器运行的窗口中。但是，如果您在浏览器中按下停止按钮，它将向您显示要重定向的 URL。单击该 URL。它将重定向到以*http://127.0.0.1*开头的内容，这不会解析，但这并不重要。将该 URL 输入回现在在笔记本页面上显示的框中，然后按 Enter。这应该授权您。

您只需要执行一次此操作；令牌将存储在名为*.cache-<username>*的文件中。如果出现问题，请删除此文件并重试。

## 讨论

Spotify API 是一个非常好的音乐数据来源。该 API 通过一个设计精良的 REST API 可访问，具有定义良好的端点，返回自描述的 JSON 文档。

[API 文档](https://developer.spotify.com/web-api/)提供了有关如何访问歌曲、艺术家和播放列表的信息，包括丰富的元信息，如专辑封面。

# 15.6 从 Spotify 收集播放列表和歌曲

## 问题

您需要为您的音乐推荐器创建一个训练集。

## 解决方案

搜索常见词以找到播放列表，并获取属于它们的歌曲。

尽管 Spotify API 非常丰富，但没有简单的方法可以获取一组公共播放列表。但您可以通过单词搜索它们。在这个配方中，我们将使用这种方法来获取一组不错的播放列表。让我们首先实现一个函数来获取与搜索词匹配的所有播放列表。代码中唯一的复杂之处在于我们需要从超时和其他错误中恢复：

```py
def find_playlists(session, w, max_count=5000):
    try:
        res = session.search(w, limit=50, type='playlist')
        while res:
            for playlist in res['playlists']['items']:
                yield playlist
                max_count -= 1
                if max_count == 0:
                    raise StopIteration
            tries = 3
            while tries > 0:
                try:
                    res = session.next(res['playlists'])
                    tries = 0
                except SpotifyException as e:
                    tries -= 1
                    time.sleep(0.2)
                    if tries == 0:
                        raise
    except SpotifyException as e:
        status = e.http_status
        if status == 404:
            raise StopIteration
        raise
```

我们将从一个单词“a”开始，获取包含该单词的 5,000 个播放列表。我们将跟踪所有这些播放列表，同时计算出现在这些播放列表标题中的单词。这样，当我们完成单词“a”后，我们可以使用出现最多的单词做同样的操作。我们可以一直这样做，直到我们有足够的播放列表：

```py
while len(playlists) < 100000:
    for word, _ in word_counts.most_common():
        if not word in words_seen:
            words_seen.add(word)
            print('word>', word)
            for playlist in find_playlists(session, word):
                if playlist['id'] in playlists:
                    dupes += 1
                elif playlist['name'] and playlist['owner']:
                    playlists[playlist['id']] = {
                      'owner': playlist['owner']['id'],
                      'name': playlist['name'],
                      'id': playlist['id'],
                    }
                    count += 1
                    for token in tokenize(playlist['name'],
                                          lowercase=True):
                        word_counts[token] += 1
            break
```

我们获取的播放列表实际上并不包含歌曲；为此，我们需要进行单独的调用。要获取播放列表的所有曲目，请使用：

```py
def track_yielder(session, playlist):
    res = session.user_playlist_tracks(playlist['owner'], playlist['id'],
          fields='items(track(id, name, artists(name, id), duration_ms)),next')
    while res:
        for track in res['items']:
            yield track['track']['id']
            res = session.next(res)
            if not res or  not res.get('items'):
                raise StopIteration
```

获取大量歌曲和播放列表可能需要相当长的时间。为了获得一些体面的结果，我们至少需要 100,000 个播放列表，但更接近一百万会更好。获取 100,000 个播放列表及其歌曲大约需要 15 个小时，这是可行的，但不是您想一遍又一遍地做的事情，所以最好保存结果。

我们将存储三个数据集。第一个包含播放列表信息本身——实际上我们不需要这个信息用于下一个配方，但检查事物时会很有用。其次，我们将把播放列表中歌曲的 ID 存储在一个大文本文件中。最后，我们将存储每首歌曲的信息。我们希望能够以动态方式查找这些详细信息，因此我们将使用 SQLite 数据库。我们将在收集歌曲信息时将结果写出，以控制内存使用：

```py
conn = sqlite3.connect('data/songs.db')
c = conn.cursor()
c.execute('CREATE TABLE songs '
          '(id text primary key, name text, artist text)')
c.execute('CREATE INDEX name_idx on songs(name)')

tracks_seen = set()
with open('data/playlists.ndjson', 'w') as fout_playlists:
    with open('data/songs_ids.txt', 'w') as fout_song_ids:
        for playlist in tqdm.tqdm(playlists.values()):
            fout_playlists.write(json.dumps(playlist) + '\n')
            track_ids = []
            for track in track_yielder(session, playlist):
                track_id = track['id']
                if not track_id:
                    continue
                if not track_id in tracks_seen:
                    c.execute("INSERT INTO songs VALUES (?, ?, ?)",
                              (track['id'], track['name'],
                               track['artists'][0]['name']))
                track_ids.append(track_id)
            fout_song_ids.write(' '.join(track_ids) + '\n')
            conn.commit()
conn.commit()
```

## 讨论

在这个配方中，我们研究了建立播放列表及其歌曲数据库。由于没有明确的方法可以从 Spotify 获取公共播放列表的平衡样本，我们采取了使用搜索界面并尝试流行关键词的方法。虽然这样做有效，但我们获取的数据集并不完全公正。

首先，我们从获取的播放列表中获取流行关键词。这确实为我们提供了与音乐相关的单词，但也很容易增加我们已经存在的偏差。如果最终我们的播放列表过多地涉及乡村音乐，那么我们的单词列表也将开始充斥着与乡村相关的单词，这反过来将使我们获取更多的乡村音乐。

另一个偏见风险是获取包含流行词的播放列表将使我们得到流行歌曲。像“最伟大”和“热门”这样的术语经常出现，会导致我们获得很多最伟大的热门歌曲；小众专辑被选中的机会较小。

# 15.7 训练音乐推荐系统

## 问题

您已经获取了大量的播放列表，但如何使用它们来训练您的音乐推荐系统呢？

## 解决方案

使用现成的 Word2vec 模型，将歌曲 ID 视为单词。

在第三章中，我们探讨了 Word2vec 模型如何将单词投影到具有良好属性的语义空间中；相似的单词最终位于同一邻域，单词之间的关系也相对一致。在第四章中，我们使用了嵌入技术构建了一个电影推荐系统。在这个配方中，我们结合了这两种方法。我们不会训练自己的模型，而是使用现成的 Word2vec 模型，但我们将使用结果来构建一个音乐推荐系统。

我们在第三章中使用的`gensim`模块还具有训练模型的可能性。它所需要的只是一个生成一系列标记的迭代器。这并不太难，因为我们将我们的播放列表存储为文件中的行，每行包含由空格分隔的歌曲 ID：

```py
class WordSplitter(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as fin:
            for line in fin:
                yield line.split()
```

之后，训练模型只需一行操作：

```py
model = gensim.models.Word2Vec(model_input, min_count=4)
```

根据前一个配方产生的歌曲/播放列表数量，这可能需要一段时间。让我们保存模型以备将来使用：

```py
with open('zoo/15/songs.word2vec', 'wb') as fout:
    model.save(fout)
```

# 15.8 使用 Word2vec 模型推荐歌曲

## 问题

如何使用您的模型根据示例预测歌曲？

## 解决方案

使用 Word2vec 距离和您的 SQLite3 歌曲数据库。

第一步是根据歌曲名称或部分名称获取一组`song_id`。`LIKE`运算符将为我们提供与搜索模式匹配的歌曲选择。但是，如今歌曲名称很少是唯一的。即使是同一位艺术家也有不同版本。因此，我们需要一种评分方式。幸运的是，我们可以使用模型的`vocab`属性——其中的记录具有*count*属性。歌曲在我们的播放列表中出现的次数越多，它就越有可能是我们要找的歌曲（或者至少是我们最了解的歌曲）：

```py
conn = sqlite3.connect('data/songs.db')
def find_song(song_name, limit=10):
    c = conn.cursor()
    c.execute("SELECT * FROM songs WHERE UPPER(name) LIKE '%"
              + song_name + "%'")
    res = sorted((x + (model.wv.vocab[x[0]].count,)
                  for x in c.fetchall() if x[0] in model.wv.vocab),
                 key=itemgetter(-1), reverse=True)
    return [*res][:limit]

for t in find_song('the eye of the tiger'):
    print(*t)
```

```py
2ZqGzZWWZXEyPxJy6N9QhG The eye of the tiger Chiara Mastroianni 39
4rr0ol3zvLiEBmep7HaHtx The Eye Of The Tiger Survivor 37
0R85QWa6KRzB8p44XXE7ky The Eye of the Tiger Gloria Gaynor 29
3GxdO4rTwVfRvLRIZFXJVu The Eye of the Tiger Gloria Gaynor 19
1W602jfZkdAsbabmJEYfFi The Eye of the Tiger Gloria Gaynor 5
6g197iis9V2HP7gvc5ZpGy I Got the Eye of the Tiger Circus Roar 5
00VQxzTLqwqBBE0BuCVeer The Eye Of The Tiger Gloria Gaynor 5
28FwycRDU81YOiGgIcxcPq The Eye of the Tiger Gloria Gaynor 5
62UagxK6LuPbqUmlygGjcU It's the Eye of the Tiger Be Cult 4
6lUHKc9qrIHvkknXIrBq6d The Eye Of The Tiger Survivor 4
```

现在我们可以选择我们真正想要的歌曲，这种情况下可能是 Survivor 的那首歌。现在开始建议歌曲。让我们的模型来做繁重的工作：

```py
similar = dict(model.most_similar([song_id]))
```

现在我们有了从歌曲 ID 到分数的查找表，我们可以很容易地扩展为实际歌曲的列表：

```py
song_ids = ', '.join(("'%s'" % x) for x in similar.keys())
c.execute("SELECT * FROM songs WHERE id in (%s)" % song_ids)
res = sorted((rec + (similar[rec[0]],) for rec in c.fetchall()),
             key=itemgetter(-1),
             reverse=True)
```

“The Eye of the Tiger”的输出是：

```py
Girls Just Wanna Have Fun Cyndi Lauper 0.9735351204872131
Enola Gay - Orchestral Manoeuvres In The Dark 0.9719518423080444
You're My Heart, You're My Soul Modern Talking 0.9589041471481323
Gold - 2003 Remastered Version Spandau Ballet 0.9566971659660339
Dolce Vita Ryan Paris 0.9553133249282837
Karma Chameleon - 2002 Remastered Version Culture Club 0.9531201720237732
Bette Davis Eyes Kim Carnes 0.9499865770339966
Walking On Sunshine Katrina & The Waves 0.9481900930404663
Maneater Daryl Hall & John Oates 0.9481032490730286
Don't You Want Me The Human League 0.9471924901008606
```

这看起来是一种不错的充满活力的 80 年代风格音乐的混合。

## 讨论

使用 Word2vec 是创建歌曲推荐系统的有效方法。与我们在第四章中所做的训练自己的模型不同，我们在这里使用了来自`gensim`的现成模型。虽然调整较少，但由于句子中的单词和播放列表中的歌曲是相当可比的，因此它效果很好。

Word2vec 通过尝试从上下文预测单词来工作。这种预测导致嵌入，使得相似的单词彼此靠近。在播放列表中对歌曲运行相同的过程意味着尝试根据播放列表中歌曲的上下文来预测歌曲。相似的歌曲最终在歌曲空间中靠近彼此。

使用 Word2vec，事实证明单词之间的关系也具有意义。分隔“queen”和“princess”之间的向量类似于分隔“king”和“prince”之间的向量。有趣的是，看看是否可以用类似的方式处理歌曲——披头士版本的滚石乐队的“Paint It Black”是什么？然而，这将要求我们以某种方式将艺术家投影到相同的空间中。
