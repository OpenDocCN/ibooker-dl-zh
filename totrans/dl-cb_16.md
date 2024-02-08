# 第十六章：生产机器学习系统

构建和训练模型是一回事；在生产系统中部署您的模型是另一回事，通常被忽视。在 Python 笔记本中运行代码很好，但不是为 web 客户端提供服务的好方法。在本章中，我们将看看如何真正开始运行。

我们将从嵌入开始。嵌入在本书的许多食谱中发挥了作用。在第三章中，我们看到了我们可以通过查看最近邻来找到相似单词的有趣事情，或者通过添加和减去嵌入向量来找到类比。在第四章中，我们使用维基百科文章的嵌入来构建一个简单的电影推荐系统。在第十章中，我们看到了我们可以将预训练图像分类网络的最终层的输出视为输入图像的嵌入，并使用此来构建反向图像搜索服务。

就像这些示例一样，我们发现真实世界的案例通常以某些实体的嵌入结束，然后我们希望从生产质量的应用程序中查询这些实体。换句话说，我们有一组图像、文本或单词，以及一个算法，为每个实体在高维空间中生成一个向量。对于一个具体的应用程序，我们希望能够查询这个空间。

我们将从一个简单的方法开始：我们将构建一个最近邻模型并将其保存到磁盘，以便在需要时加载。然后我们将看看如何使用 Postgres 达到相同的目的。

我们还将探讨使用微服务作为一种使用 Flask 作为 web 服务器和 Keras 的保存和加载模型功能来暴露机器学习模型的方法。

以下笔记本可用于本章：

```py
16.1 Simple Text Generation
16.2 Prepare Keras Model for TensorFlow Serving
16.3 Prepare Model for iOS
```

# 16.1 使用 Scikit-Learn 的最近邻算法进行嵌入

## 问题

如何快速提供嵌入模型的最接近匹配项？

## 解决方案

使用 scikit-learn 的最近邻算法并将模型保存到文件中。我们将继续从第四章的代码开始，我们在那里创建了一个电影预测模型。在我们运行完所有内容后，我们将归一化值并拟合一个最近邻模型：

```py
movie = model.get_layer('movie_embedding')
movie_weights = movie.get_weights()[0]
movie_lengths = np.linalg.norm(movie_weights, axis=1)
normalized_movies = (movie_weights.T / movie_lengths).T
nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(
    normalized_movies)
with open('data/movie_model.pkl', 'wb') as fout:
    pickle.dump({
        'nbrs': nbrs,
        'normalized_movies': normalized_movies,
        'movie_to_idx': movie_to_idx
    }, fout)
```

然后稍后可以再次加载模型：

```py
with open('data/movie_model.pkl', 'rb') as fin:
    m = pickle.load(fin)
movie_names = [x[0] for x in sorted(movie_to_idx.items(),
               key=lambda t:t[1])]
distances, indices = m['nbrs'].kneighbors(
    [m['normalized_movies'][m['movie_to_idx']['Rogue One']]])
for idx in indices[0]:
    print(movie_names[idx])
```

```py
Rogue One
Prometheus (2012 film)
Star Wars: The Force Awakens
Rise of the Planet of the Apes
Star Wars sequel trilogy
Man of Steel (film)
Interstellar (film)
Superman Returns
The Dark Knight Trilogy
Jurassic World
```

## 讨论

生产机器学习模型的最简单方法是在训练完成后将其保存到磁盘，然后在需要时加载。所有主要的机器学习框架都支持这一点，包括我们在本书中使用的 Keras 和 scikit-learn。

如果您控制内存管理，这个解决方案非常好。然而，在生产 web 服务器中，通常情况并非如此，当 web 请求到来时，如果必须将大型模型加载到内存中，延迟显然会受到影响。

# 16.2 使用 Postgres 存储嵌入

## 问题

您想使用 Postgres 存储嵌入。

## 解决方案

使用 Postgres 的`Cube`扩展。

`Cube`扩展允许处理高维数据，但需要先启用它：

```py
CREATE EXTENSION cube;
```

完成后，我们可以创建一个表和相应的索引。为了也能够在`movie_name`字段上进行搜索，我们还将在`movie_name`字段上创建一个文本索引：

```py
DROP TABLE IF EXISTS movie;
CREATE TABLE movie (
               movie_name TEXT PRIMARY KEY,
               embedding FLOAT[] NOT NULL DEFAULT '{}'
);
CREATE INDEX movie_embedding ON movie USING gin(embedding);
CREATE INDEX movie_movie_name_pattern
    ON movie USING btree(lower(movie_name) text_pattern_ops);
```

## 讨论

Postgres 是一个免费的数据库，非常强大，其中一个原因是有大量可用的扩展。其中一个模块是`cube`模块。顾名思义，它最初是为了将三维坐标作为原始数据可用，但后来已扩展到可以索引高达 100 维的数组。

Postgres 有许多扩展，值得探索，特别是对于处理大量数据的任何人。特别是，在经典 SQL 表中以数组和 JSON 文档的形式存储较少结构化的数据的能力在原型设计时非常方便。

# 16.3 填充和查询存储在 Postgres 中的嵌入

## 问题

您能够在 Postgres 中存储我们的模型和查询结果吗？

## 解决方案

使用`psycopg2`从 Python 连接到 Postgres。

通过给定的用户名/密码/数据库/主机组合，我们可以轻松地使用 Python 连接到 Postgres：

```py
connection_str = "dbname='%s' user='%s' password='%s' host='%s'"
conn = psycopg2.connect(connection_str % (DB_NAME, USER, PWD, HOST))

```

插入我们之前构建的模型与 Python 中的任何其他 SQL 操作一样，只是我们需要将我们的`numpy`数组转换为 Python 列表：

```py
with conn.cursor() as cursor:
    for movie, embedding in zip(movies, normalized_movies):
        cursor.execute('INSERT INTO movie (movie_name, embedding)'
                       ' VALUES (%s, %s)',
               (movie[0], embedding.tolist()))
conn.commit()
```

完成后，我们可以查询这些值。在这种情况下，我们取（部分）电影标题，找到该电影的最佳匹配，并返回最相似的电影：

```py
def recommend_movies(conn, q):
    with conn.cursor() as cursor:
        cursor.execute('SELECT movie_name, embedding FROM movie'
                       '    WHERE lower(movie_name) LIKE %s'
                       '    LIMIT 1',
                       ('%' + q.lower() + '%',))
        if cursor.rowcount == 0:
            return []
        movie_name, embedding = cursor.fetchone()
        cursor.execute('SELECT movie_name, '
                       '       cube_distance(cube(embedding), '
                       '                     cube(%s)) as distance '
                       '    FROM movie'
                       '    ORDER BY distance'
                       '    LIMIT 5',
                       (embedding,))
        return list(cursor.fetchall())
```

## 讨论

将嵌入模型存储在 Postgres 数据库中允许我们直接查询它，而无需在每个请求中加载模型，因此当我们想要从 Web 服务器使用这样的模型时，这是一个很好的解决方案——特别是当我们的 Web 设置一开始就是基于 Postgres 的时候。

在支持您网站的数据库服务器上运行模型或模型的结果具有额外的优势，您可以无缝地混合排名组件。我们可以轻松地扩展这个示例的代码，将新鲜度评分包含在我们的电影表中，从那时起，我们可以使用这些信息来帮助对返回的电影进行排序。但是，如果评分和相似性距离来自不同的来源，我们要么必须手动进行内存连接，要么返回不完整的结果。

# 16.4 在 Postgres 中存储高维模型

## 问题

如何在 Postgres 中存储具有超过 100 个维度的模型？

## 解决方案

使用降维技术。

假设我们想要加载谷歌预训练的 Word2vec 模型，我们在第三章中使用了这个模型，加载到 Postgres 中。由于 Postgres 的`cube`扩展（参见 Recipe 16.2）限制了它将索引的维度数量为 100，我们需要做一些处理以使其适应。使用奇异值分解（SVD）来降低维度——这是我们在 Recipe 10.4 中遇到的一种技术——是一个不错的选择。让我们像以前一样加载 Word2vec 模型：

```py
model = gensim.models.KeyedVectors.load_word2vec_format(
    MODEL, binary=True)
```

每个单词的归一化向量存储在`syn0norm`属性中，因此我们可以在其上运行 SVD。这需要一点时间：

```py
svd = TruncatedSVD(n_components=100, random_state=42,
                   n_iter=40)
reduced = svd.fit_transform(model.syn0norm)
```

我们需要重新归一化向量：

```py
reduced_lengths = np.linalg.norm(reduced, axis=1)
normalized_reduced = reduced.T / reduced_lengths).T
```

现在我们可以看相似性：

```py
def most_similar(norm, positive):
    vec = norm[model.vocab[positive].index]
    dists = np.dot(norm, vec)
    most_extreme = np.argpartition(-dists, 10)[:10]
    res = ((model.index2word[idx], dists[idx]) for idx in most_extreme)
    return list(sorted(res, key=lambda t:t[1], reverse=True))
for word, score in most_similar(normalized_reduced, 'espresso'):
    print(word, score)
```

```py
espresso 1.0
cappuccino 0.856463080029
chai_latte 0.835657488972
latte 0.800340435865
macchiato 0.798796776324
espresso_machine 0.791469456128
Lavazza_coffee 0.790783985201
mocha 0.788645681469
espressos 0.78424218748
martini 0.784037414689
```

结果看起来仍然合理，但并不完全相同。最后一个条目，马提尼，出现在一系列含有咖啡因提神饮料的列表中有些出乎意料。

## 讨论

Postgres 的`cube`扩展很棒，但带有一个警告，它仅适用于具有 100 个或更少元素的向量。文档有用地解释了这个限制：“为了让人们更难破坏事物，立方体的维度数量限制为 100。”绕过这个限制的一种方法是重新编译 Postgres，但这只是一个选项，如果您直接控制您的设置。此外，随着数据库的新版本发布，您需要继续这样做。

在将我们的向量插入数据库之前降低维度可以很容易地使用`TruncatedSVD`类来完成。在这个示例中，我们使用了 Word2vec 数据集中的整个单词集，这导致了一些精度的损失。如果我们不仅降低输出的维度，还减少术语的数量，我们可以做得更好。然后 SVD 可以找到我们提供的数据中最重要的维度，而不是所有数据。这甚至可以通过泛化一点并掩盖原始输入数据中的数据不足来帮助。

# 16.5 使用 Python 编写微服务

## 问题

您想编写并部署一个简单的 Python 微服务。

## 解决方案

使用 Flask 构建一个最小的 Web 应用程序，根据 REST 请求返回一个 JSON 文档。

首先我们需要一个 Flask Web 服务器：

```py
app = Flask(__name__)
```

然后我们定义我们想要提供的服务。例如，我们将接收一张图片并返回图片的大小。我们期望图片是`POST`请求的一部分。如果我们没有收到`POST`请求，我们将返回一个简单的 HTML 表单，这样我们就可以在没有客户端的情况下测试服务。`@app.route`装饰指定`return_size`处理根目录的任何请求，支持`GET`和`POST`：

```py
@app.route('/', methods=['GET', 'POST'])
def return_size():
  if request.method == 'POST':
    file = request.files['file']
    if file:
      image = Image.open(file)
      width, height = image.size
      return jsonify(results={'width': width, 'height': height})
  return '''
 <h1>Upload new File</h1>
 <form action="" method=post enctype=multipart/form-data>
 <p><input type=file name=file>
 <input type=submit value=Upload>
 </form>
 '''
```

现在我们只需要在一个端口上运行服务器：

```py
app.run(port=5050, host='0.0.0.0')
```

## 讨论

REST 最初是作为一个完整的资源管理框架，为系统中的所有资源分配 URL，并让客户端与 HTTP 谓词的整个范围进行交互，从`PUT`到`DELETE`。就像许多 API 一样，在这个示例中我们放弃了所有这些，只在一个处理程序上定义了一个`GET`方法，触发我们的 API 并返回一个 JSON 文档。

我们在这里开发的服务当然相当琐碎；仅仅为了获取图像大小而拥有一个微服务可能有点过头了。在下一个示例中，我们将探讨如何使用这种方法来提供先前开发的机器学习模型的结果。

# 16.6 使用微服务部署 Keras 模型

## 问题

您想要将 Keras 模型部署为独立服务。

## 解决方案

将您的 Flask 服务器扩展到转发请求到预训练的 Keras 模型。

这个示例是在第十章中的示例构建的，我们从维基百科下载了成千上万张图片，并将它们馈送到一个预训练的图像识别网络中，得到一个描述每张图片的 2,048 维向量。我们将在这些向量上拟合一个最近邻模型，以便我们可以快速找到最相似的图像，给定一个向量。

第一步是加载腌制的图像名称和最近邻模型，并实例化用于图像识别的预训练模型：

```py
with open('data/image_similarity.pck', 'rb') as fin:
    p = pickle.load(fin)
    image_names = p['image_names']
    nbrs = p['nbrs']
base_model = InceptionV3(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input,
              outputs=base_model.get_layer('avg_pool').output)
```

我们现在可以通过更改`if file:`后的代码部分来修改如何处理传入的图像。我们将调整图像大小到模型的目标大小，规范化数据，运行预测，并找到最近的邻居：

```py
      img = Image.open(file)
      target_size = int(max(model.input.shape[1:]))
      img = img.resize((target_size, target_size), Image.ANTIALIAS)
      pre_processed = preprocess_input(
          np.asarray([image.img_to_array(img)]))
      vec = model.predict(pre_processed)
      distances, indices = nbrs.kneighbors(vec)
      res = [{'distance': dist,
              'image_name': image_names[idx]}
             for dist, idx in zip(distances[0], indices[0])]
      return jsonify(results=res)
```

给它一张猫的图片，你应该看到从维基百科图片中抽样出大量猫的图片——还有一张孩子们玩家用电脑的照片。

## 讨论

通过在启动时加载模型，然后在图像进入时馈送图像，我们可以减少我们遵循本节第一个示例方法所获得的延迟。我们在这里有效地链接了两个模型，预训练的图像识别网络和最近邻分类器，并将组合导出为一个服务。

# 16.7 从 Web 框架调用微服务

## 问题

您想要从 Django 调用一个微服务。

## 解决方案

使用`requests`调用微服务同时处理 Django 请求。我们可以按照以下示例的方式进行：

```py
def simple_view(request):
    d = {}
    update_date(request, d)
    if request.FILES.get('painting'):
        data = request.FILES['painting'].read()
        files = {'file': data}
        reply = requests.post('http://localhost:5050',
                              files=files).json()
        res = reply['results']
        if res:
            d['most_similar'] = res[0]['image_name']
    return render(request, 'template_path/template.html', d)
```

## 讨论

这里的代码来自 Django 请求处理程序，但在其他 Web 框架中看起来应该非常相似，即使是基于 Python 以外的语言的框架。

这里的关键是我们将 Web 框架的会话管理与微服务的会话管理分开。这样我们就知道在任何给定时间只有一个模型实例，这使得延迟和内存使用可预测。

`Requests`是一个用于发起`HTTP`调用的简单模块。尽管它不支持发起异步调用。在这个示例的代码中，这并不重要，但如果我们需要调用多个服务，我们希望能够并行进行。有许多选项可供选择，但它们都遵循一种模式，即在请求开始时向后端发起调用，进行我们需要的处理，然后在需要结果时等待未完成的请求。这是使用 Python 构建高性能系统的良好设置。

# 16.8 TensorFlow seq2seq 模型

## 问题

如何将 seq2seq 聊天模型投入生产？

## 解决方案

运行一个带有输出捕获钩子的 TensorFlow 会话。

Google 发布的`seq2seq`模型是一种非常好的快速开发序列到序列模型的方式，但默认情况下推断阶段只能使用`stdin`和`stdout`运行。通过这种方式从我们的微服务调用是完全可能的，但这意味着我们将在每次调用时承担加载模型的延迟成本。

更好的方法是手动实例化模型并使用钩子捕获输出。第一步是从检查点目录中恢复模型。我们需要加载模型和模型配置。模型将输入`source_tokens`（即聊天提示），我们将使用批量大小为 1，因为我们将以交互方式进行：

```py
checkpoint_path = tf.train.latest_checkpoint(model_path)
train_options = training_utils.TrainOptions.load(model_path)
model_cls = locate(train_options.model_class) or \
  getattr(models, train_options.model_class)
model_params = train_options.model_params
model = model_cls(
    params=model_params,
    mode=tf.contrib.learn.ModeKeys.INFER)
source_tokens_ph = tf.placeholder(dtype=tf.string, shape=(1, None))
source_len_ph = tf.placeholder(dtype=tf.int32, shape=(1,))
model(
  features={
    "source_tokens": source_tokens_ph,
    "source_len": source_len_ph
  },
  labels=None,
  params={
  }
)
```

下一步是设置允许我们将数据输入模型的 TensorFlow 会话。这都是相当标准的内容（应该让我们更加欣赏像 Keras 这样的框架）：

```py
  saver = tf.train.Saver()
  def _session_init_op(_scaffold, sess):
      saver.restore(sess, checkpoint_path)
      tf.logging.info("Restored model from %s", checkpoint_path)
  scaffold = tf.train.Scaffold(init_fn=_session_init_op)
  session_creator = tf.train.ChiefSessionCreator(scaffold=scaffold)
  sess = tf.train.MonitoredSession(
      session_creator=session_creator,
      hooks=[DecodeOnce({}, callback_func=_save_prediction_to_dict)])
  return sess, source_tokens_ph, source_len_pht
```

我们现在配置了一个带有`DecodeOnce`钩子的 TensorFlow 会话，它是一个实现推断任务基本功能的类，但在完成后调用提供的`callback`函数返回实际结果。

在*seq2seq_server.py*的代码中，我们可以使用这个来处理 HTTP 请求，如下所示：

```py
@app.route('/', methods=['GET'])
def handle_request():
  input = request.args.get('input', '')
  if input:
    tf.reset_default_graph()
    source_tokens = input.split() + ["SEQUENCE_END"]
    session.run([], {
        source_tokens_ph: [source_tokens],
        source_len_ph: [len(source_tokens)]
      })
    return prediction_dict.pop(_tokens_to_str(source_tokens))
```

这将让我们处理来自简单 Web 服务器的 seq2seq 调用。

## 讨论

在这个配方中，我们将数据输入到 seq2seq TensorFlow 模型的方式并不是很漂亮，但它是有效的，并且在性能方面比使用`stdin`和`stdout`要好得多。希望这个库的即将推出的版本将为我们提供一个更好的方式来在生产中使用这些模型，但目前这样做就可以了。

# 16.9 在浏览器中运行深度学习模型

## 问题

如何在没有服务器的情况下运行深度学习 Web 应用程序？

## 解决方案

使用 Keras.js 在浏览器中运行模型。

在浏览器中运行深度学习模型听起来很疯狂。深度学习需要大量的处理能力，我们都知道 JavaScript 很慢。但事实证明，您可以在浏览器中以相当快的速度运行模型，并使用 GPU 加速。[Keras.js](https://transcranial.github.io/keras-js/#)有一个工具，可以将 Keras 模型转换为 JavaScript 运行时可以处理的内容，并使用 WebGL 来让 GPU 帮助处理。这是一个令人惊叹的工程成就，并且配备了一些令人印象深刻的演示。让我们尝试在我们自己的模型上尝试一下。

笔记本`16.1 简单文本生成`取自 Keras 示例目录，并基于尼采的著作训练了一个简单的文本生成模型。训练后，我们保存模型：

```py
model.save('keras_js/nietzsche.h5')
with open('keras_js/chars.js', 'w') as fout:
    fout.write('maxlen = ' + str(maxlen) + '\n')
    fout.write('num_chars = ' + str(len(chars)) + '\n')
    fout.write('char_indices = ' + json.dumps(char_indices, indent=2) + '\n')
    fout.write('indices_char = ' + json.dumps(indices_char, indent=2) + '\n')
```

现在我们需要将 Keras 模型转换为 Keras.js 格式。首先获取转换代码：

```py
git clone https://github.com/transcranial/keras-js.git
```

现在打开一个 shell，在保存模型的目录中执行：

```py
python <*git-root*>/keras-js/python/encoder.py nietzsche.h5

```

这将给您一个*nietzsche.bin*文件。

下一步是从网页中使用这个文件。

我们将在*nietzsche.html*文件中执行此操作，您将在*deep_learning_cookbook*存储库的*keras_js*目录中找到它。让我们来看看。它以加载 Keras.js 库和我们从 Python 保存的变量的代码开始：

```py
<script src="https://unpkg.com/keras-js"></script>
<script src="chars.js"></script>
```

在底部，我们有一个非常简单的 HTML 代码，让用户输入一些文本，然后按下按钮以尼采式的方式扩展文本：

```py
<textarea cols="60" rows="4" id="textArea">
   i am all for progress, it is
</textarea><br/>
<button onclick="runModel(250)" disabled id="buttonGo">Go!</button>
```

现在让我们加载模型，加载完成后，启用当前禁用的按钮`buttonGo`：

```py
const model = new KerasJS.Model({
      filepath: 'sayings.bin',
      gpu: true
    })

    model.ready().then(() => {
      document.getElementById("buttonGo").disabled = false
    })
```

在`runModel`中，我们首先需要使用之前导入的`char_indices`对文本数据进行独热编码：

```py
    function encode(st) {
      var x = new Float32Array(num_chars * st.length);
      for(var i = 0; i < st.length; i++) {
        idx = char_indices[ch = st[i]];
        x[idx + i * num_chars] = 1;
      }
      return x;
    };
```

现在我们可以运行模型：

```py
return model.predict(inputData).then(outputData => {
    ...
    ...
  })
```

`outputData`变量将包含我们词汇表中每个字符的概率分布。理解这一点最简单的方法是选择具有最高概率的字符：

```py
      var maxIdx = -1;
      var maxVal = 0.0;
      for (var idx = 0; idx < output.length; idx ++) {
        if (output[idx] > maxVal) {
          maxVal = output[idx];
          maxIdx = idx;
        }
      }
```

现在我们只需将该字符添加到我们到目前为止的内容中，并再次执行相同的操作：

```py
     var nextChar = indices_char["" + maxIdx];
      document.getElementById("textArea").value += nextChar;
      if (steps > 0) {
        runModel(steps - 1);
      }
```

## 讨论

能够直接在浏览器中运行模型为生产提供了全新的可能性。这意味着您不需要服务器来执行实际计算，并且使用 WebGL，您甚至可以免费获得 GPU 加速。查看[*https://transcranial.github.io/keras-js*](https://transcranial.github.io/keras-js)上的有趣演示。

这种方法存在一些限制。为了使用 GPU，Keras.js 使用 WebGL 2.0。不幸的是，目前并非所有浏览器都支持这一点。此外，张量被编码为 WebGL 纹理，其大小受限。实际限制取决于您的浏览器和硬件。当然，您可以退回到仅使用 CPU，但这意味着在纯 JavaScript 中运行。

第二个限制是模型的大小。生产质量的模型通常有几十兆字节的大小，当它们一次加载到服务器上时，这一点根本不是问题，但当它们需要发送到客户端时可能会出现问题。

###### 注意

*encoder.py*脚本有一个名为`--quantize`的标志，它将模型的权重编码为 8 位整数。这将减少模型大小 75%，但意味着权重将不太精确，这可能会影响预测准确性。

# 16.10 使用 TensorFlow Serving 运行 Keras 模型

## 问题

如何使用谷歌最先进的服务器运行 Keras 模型？

## 解决方案

转换模型并调用 TensorFlow Serving 工具包以写出模型规范，以便您可以使用 TensorFlow Serving 运行它。

TensorFlow Serving 是 TensorFlow 平台的一部分；根据谷歌的说法，它是一个灵活、高性能的机器学习模型服务系统，专为生产环境设计。

将一个 TensorFlow 模型写成 TensorFlow Serving 可以使用的方式有些复杂。为了使其正常工作，Keras 模型需要更多的调整。原则上，只要模型只有一个输入和一个输出，就可以使用任何模型——这是 TensorFlow Serving 的限制之一。另一个限制是 TensorFlow Serving 只支持 Python 2.7。

首先要做的是将模型重新创建为仅用于测试的模型。模型在训练和测试期间的行为不同。例如，`Dropout`层在训练时只会随机丢弃神经元，而在测试时会使用所有神经元。Keras 会将这些隐藏在用户之外，将学习阶段作为额外变量传递。如果您看到错误提示缺少输入的某些内容，这可能是原因。我们将学习阶段设置为`0`（false），并从我们的字符 CNN 模型中提取配置和权重：

```py
K.set_learning_phase(0)
char_cnn = load_model('zoo/07.2 char_cnn_model.h5')
config = char_cnn.get_config()
if not 'config' in config:
    config = {'config': config,
              'class_name': 'Model'}

weights = char_cnn.get_weights()
```

此时可能有必要对模型进行预测，以便稍后查看它仍然有效：

```py
tweet = ("There's a house centipede in my closet and "
         "since Ryan isn't here I have to kill it....")
encoded = np.zeros((1, max_sequence_len, len(char_to_idx)))
for idx, ch in enumerate(tweet):
    encoded[0, idx, char_to_idx[ch]] = 1

res = char_cnn.predict(encoded)
emojis[np.argmax(res)]
```

```py
u'\ude03'
```

然后我们可以重新构建模型：

```py
new_model = model_from_config(config)
new_model.set_weights(weights)
```

为了使模型运行，我们需要为 TensorFlow Serving 提供输入和输出规范：

```py
input_info = utils.build_tensor_info(new_model.inputs[0])
output_info = utils.build_tensor_info(new_model.outputs[0])
prediction_signature = signature_def_utils.build_signature_def(
          inputs={'input': input_info},
          outputs={'output': output_info},
          method_name=signature_constants.PREDICT_METHOD_NAME)
```

然后我们可以构建`builder`对象来定义我们的处理程序并写出定义：

```py
outpath = 'zoo/07.2 char_cnn_model.tf_model/1'
shutil.rmtree(outpath)

legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
builder = tf.saved_model.builder.SavedModelBuilder(outpath)
builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
           'emoji_suggest': prediction_signature,
      },
      legacy_init_op=legacy_init_op)
builder.save()
```

现在我们运行服务器：

```py
tensorflow_model_server \
    --model_base_path="char_cnn_model.tf_model/" \
    --model_name="char_cnn_model"
```

您可以直接从谷歌获取二进制文件，也可以从源代码构建——有关详细信息，请参阅[安装说明](https://www.tensorflow.org/serving/setup)。

让我们看看我们是否可以从 Python 调用模型。我们将实例化一个预测请求并使用`grpc`进行调用：

```py
request = predict_pb2.PredictRequest()
request.model_spec.name = 'char_cnn_model'
request.model_spec.signature_name = 'emoji_suggest'
request.inputs['input'].CopyFrom(tf.contrib.util.make_tensor_proto(
    encoded.astype('float32'), shape=[1, max_sequence_len, len(char_to_idx)]))

channel = implementations.insecure_channel('localhost', 8500)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
result = stub.Predict(request, 5)
```

获取实际预测的表情符号：

```py
response = np.array(result.outputs['output'].float_val)
prediction = np.argmax(response)
emojis[prediction]
```

## 讨论

TensorFlow Serving 是谷歌认可的将模型投入生产的方式，但与启动自定义 Flask 服务器并自行处理输入和输出相比，使用它与 Keras 模型有些复杂。

尽管有优势。首先，由于不是自定义的，这些服务器都表现一致。此外，它是一个工业强度的服务器，支持版本控制，并可以直接从多个云提供商加载模型。

# 16.11 在 iOS 上使用 Keras 模型

## 问题

您希望在 iOS 上的移动应用程序中使用在桌面上训练的模型。

## 解决方案

使用 CoreML 将您的模型转换并直接从 Swift 与之通信。

###### 注意

本文介绍了如何为 iOS 构建应用程序，因此您需要安装了 Xcode 的 Mac 来运行示例。此外，由于示例使用相机进行检测，因此您还需要具有相机的 iOS 设备来尝试它。

首先要做的是转换模型。不幸的是，苹果的代码只支持 Python 2.7，并且在支持最新版本的`tensorflow`和`keras`方面似乎有点滞后，所以我们将设置特定版本。打开一个 shell 来设置 Python 2.7 并输入正确的要求，然后输入：

```py
virtualenv venv2
source venv2/bin/activate
pip install coremltools
pip install h5py
pip install keras==2.0.6
pip install tensorflow==1.2.1
```

然后启动 Python 并输入：

```py
from keras.models import load_model
import coremltools
```

加载先前保存的模型和标签：

```py
keras_model = load_model('zoo/09.3 retrained pet recognizer.h5')
class_labels = json.load(open('zoo/09.3 pet_labels.json'))
```

然后转换模型：

```py
coreml_model = coremltools.converters.keras.convert(
    keras_model,
    image_input_names="input_1",
    class_labels=class_labels,
    image_scale=1/255.)
coreml_model.save('zoo/PetRecognizer.mlmodel')
```

###### 提示

您也可以跳过这一步，在*zoo*目录中使用*.mlmodel*文件进行工作。

现在开始 Xcode，创建一个新项目，并将*PetRecognizer.mlmodel*文件拖到项目中。Xcode 会自动导入模型并使其可调用。让我们识别一些宠物！

苹果在其网站上有一个示例项目[链接](https://apple.co/2HPUHOW)，使用了标准的图像识别网络。下载这个项目，解压缩它，然后用 Xcode 打开它。

在项目概述中，您应该看到一个名为*MobileNet.mlmodel*的文件。删除它，然后将*PetRecognizer.mlmodel*文件拖到原来*MobileNet.mlmodel*的位置。现在打开*ImageClassificationViewController.swift*，并将任何`MobileNet`的出现替换为`PetRecognizer`。

现在您应该能够像以前一样运行该应用程序，但使用新模型和输出类。

## 讨论

在 iOS 应用程序中使用 Keras 模型非常简单，至少如果我们坚持使用苹果 SDK 提供的示例的话。尽管这项技术相当新，但在那里没有太多与苹果示例大不相同的可用示例。此外，CoreML 仅适用于苹果操作系统，仅适用于 iOS 11 或更高版本或 macOS 10.13 或更高版本。
