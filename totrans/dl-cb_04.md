# 第4章。基于维基百科外部链接构建推荐系统

推荐系统传统上是根据用户先前收集的评分进行训练的。我们希望预测用户的评分，因此从历史评分开始似乎是一个自然的选择。然而，这要求我们在开始之前有一个大量的评分集，并且不允许我们对尚未评分的新项目做出良好的工作。此外，我们故意忽略了我们对项目的元信息。

在本章中，您将探索如何仅基于维基百科的外部链接构建一个简单的电影推荐系统。您将首先从维基百科中提取一个训练集，然后基于这些链接训练嵌入。然后，您将实现一个简单的支持向量机分类器来提供建议。最后，您将探索如何使用新训练的嵌入来预测电影的评分。

本章中的代码可以在这些笔记本中找到：

```py
04.1 Collect movie data from Wikipedia
04.2 Build a recommender system based on outgoing Wikipedia links
```

# 4.1 收集数据

## 问题

您希望获得一个特定领域的训练数据集，比如电影。

## 解决方案

解析维基百科转储文件并仅提取电影页面。

###### 注意

本配方中的代码展示了如何从维基百科获取和提取训练数据，这是一个非常有用的技能。然而，下载和处理完整的转储文件需要相当长的时间。笔记本文件夹的*data*目录包含了预先提取的前10000部电影，我们将在本章的其余部分中使用，因此您不需要运行本配方中的步骤。

让我们从维基百科下载最新的转储文件开始。您可以使用您喜欢的浏览器轻松完成这一操作，如果您不需要最新版本，您可能应该选择附近的镜像。但您也可以通过编程方式完成。以下是获取最新转储页面的方法：

```py
index = requests.get('https://dumps.wikimedia.org/backup-index.html').text
soup_index = BeautifulSoup(index, 'html.parser')
dumps = [a['href'] for a in soup_index.find_all('a')
             if a.has_attr('href') and a.text[:-1].isdigit()]
```

我们现在将浏览转储文件，并找到最新的已完成处理的文件：

```py
for dump_url in sorted(dumps, reverse=True):
    print(dump_url)
    dump_html = index = requests.get(
        'https://dumps.wikimedia.org/enwiki/' + dump_url).text
    soup_dump = BeautifulSoup(dump_html, 'html.parser')
    pages_xml = [a['href'] for a in soup_dump.find_all('a')
                 if a.has_attr('href')
                 and a['href'].endswith('-pages-articles.xml.bz2')]
    if pages_xml:
        break
    time.sleep(0.8)
```

注意睡眠以保持在维基百科的速率限制之下。现在让我们获取转储文件：

```py
wikipedia_dump = pages_xml[0].rsplit('/')[-1]
url = url = 'https://dumps.wikimedia.org/' + pages_xml[0]
path = get_file(wikipedia_dump, url)
path
```

我们检索到的转储文件是一个bz2压缩的XML文件。我们将使用`sax`来解析维基百科的XML。我们对`<title>`和`<page>`标签感兴趣，因此我们的`Content​Handler`看起来像这样：

```py
class WikiXmlHandler(xml.sax.handler.ContentHandler):
    def __init__(self):
        xml.sax.handler.ContentHandler.__init__(self)
        self._buffer = None
        self._values = {}
        self._movies = []
        self._curent_tag = None

    def characters(self, content):
        if self._curent_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        if name in ('title', 'text'):
            self._curent_tag = name
            self._buffer = []

    def endElement(self, name):
        if name == self._curent_tag:
            self._values[name] = ' '.join(self._buffer)

        if name == 'page':
            movie = process_article(**self._values)
            if movie:
                self._movies.append(movie)
```

对于每个`<page>`标签，这将收集标题和文本内容到`self._values`字典中，并使用收集到的值调用`process_article`。

尽管维基百科最初是一个超链接文本型百科全书，但多年来它已经发展成一个更结构化的数据转储。其中一种方法是让页面链接回所谓的*分类页面*。这些链接起到标签的作用。电影《飞越疯人院》的页面链接到“1975年电影”分类页面，因此我们知道这是一部1975年的电影。不幸的是，并没有仅仅针对电影的分类页面。幸运的是，有一个更好的方法：维基百科模板。

模板最初是一种确保包含相似信息的页面以相同方式呈现该信息的方法。“信息框”模板对数据处理非常有用。它不仅包含适用于页面主题的键/值对列表，还有一个类型。其中之一是“电影”，这使得提取所有电影的任务变得更加容易。

对于每部电影，我们想要提取名称、外部链接以及，仅仅是因为我们可以，存储在信息框中的属性。名为`mwparserfromhell`的工具在解析维基百科时表现得相当不错：

```py
def process_article(title, text):
    rotten = [(re.findall('\d\d?\d?%', p),
        re.findall('\d\.\d\/\d+|$', p), p.lower().find('rotten tomatoes'))
        for p in text.split('\n\n')]
    rating = next(((perc[0], rating[0]) for perc, rating, idx in rotten
        if len(perc) == 1 and idx > -1), (None, None))
    wikicode = mwparserfromhell.parse(text)
    film = next((template for template in wikicode.filter_templates()
                 if template.name.strip().lower() == 'infobox film'),
                 None)
    if film:
        properties = {param.name.strip_code().strip():
                      param.value.strip_code().strip()
                      for param in film.params
                      if param.value.strip_code().strip()
                     }
        links = [x.title.strip_code().strip()
                 for x in wikicode.filter_wikilinks()]
        return (title, properties, links) + rating
```

现在我们可以将bzipped转储文件输入解析器：

```py
parser = xml.sax.make_parser()
handler = WikiXmlHandler()
parser.setContentHandler(handler)
for line in subprocess.Popen(['bzcat'],
                             stdin=open(path),
                             stdout=subprocess.PIPE).stdout:
  try:
    parser.feed(line)
  except StopIteration:
    break
```

最后，让我们保存结果，这样下次我们需要数据时就不必再处理几个小时：

```py
with open('wp_movies.ndjson', 'wt') as fout:
  for movie in handler._movies:
    fout.write(json.dumps(movie) + '\n')
```

## 讨论

维基百科不仅是回答几乎任何人类知识领域问题的重要资源；它也是许多深度学习实验的起点。了解如何解析转储文件并提取相关部分是许多项目中有用的技能。

13 GB的数据转储是相当大的下载。解析维基百科标记语言带来了自己的挑战：这种语言多年来有机地发展，似乎没有强大的基础设计。但是随着今天快速的连接和一些出色的开源库来帮助解析，这一切都变得相当可行。

在某些情况下，维基百科API可能更合适。这个对维基百科的REST接口允许您以多种强大的方式搜索和查询，并且只获取您需要的文章。考虑到速率限制，以这种方式获取所有电影将需要很长时间，但对于较小的领域来说，这是一个选择。

如果您最终要为许多项目解析维基百科，那么首先将转储导入到像Postgres这样的数据库中可能是值得的，这样您就可以直接查询数据集。

# 4.2 训练电影嵌入

## 问题

如何使用实体之间的链接数据生成建议，比如“如果您喜欢这个，您可能也对那个感兴趣”？

## 解决方案

使用一些元信息作为连接器来训练嵌入。这个示例建立在之前的示例之上，使用了那里提取的电影和链接。为了使数据集变得更小且更少噪音，我们将仅使用维基百科上受欢迎程度确定的前10,000部电影。

我们将外链视为连接器。这里的直觉是链接到同一页面的电影是相似的。它们可能有相同的导演或属于相同的类型。随着模型的训练，它不仅学习哪些电影相似，还学习哪些链接相似。这样它可以泛化并发现指向1978年的链接与指向1979年的链接具有相似的含义，从而有助于电影的相似性。

我们将从计算外链开始，这是一个快速查看我们是否合理的方法：

```py
link_counts = Counter()
for movie in movies:
    link_counts.update(movie[2])
link_counts.most_common(3)
```

```py
[(u'Rotten Tomatoes', 9393),
 (u'Category:English-language films', 5882),
 (u'Category:American films', 5867)]
```

我们模型的任务是确定某个链接是否可以在电影的维基百科页面上找到，因此我们需要提供标记的匹配和不匹配示例。我们只保留至少出现三次的链接，并构建所有有效的（链接，电影）对的列表，我们将其存储以供以后快速查找。我们将保留相同的便于以后快速查找：

```py
top_links = [link for link, c in link_counts.items() if c >= 3]
link_to_idx = {link: idx for idx, link in enumerate(top_links)}
movie_to_idx = {movie[0]: idx for idx, movie in enumerate(movies)}
pairs = []
for movie in movies:
    pairs.extend((link_to_idx[link], movie_to_idx[movie[0]])
                  for link in movie[2] if link in link_to_idx)
pairs_set = set(pairs)
```

我们现在准备介绍我们的模型。从图表上看，我们将`link_id`和`movie_id`作为数字输入到它们各自的嵌入层中。嵌入层将为每个可能的输入分配一个大小为`embedding_size`的向量。然后我们将这两个向量的点积设置为我们模型的输出。模型将学习权重，使得这个点积接近标签。然后这些权重将把电影和链接投影到一个空间中，使得相似的电影最终位于相似的位置：

```py
def movie_embedding_model(embedding_size=30):
    link = Input(name='link', shape=(1,))
    movie = Input(name='movie', shape=(1,))
    link_embedding = Embedding(name='link_embedding',
        input_dim=len(top_links), output_dim=embedding_size)(link)
    movie_embedding = Embedding(name='movie_embedding',
        input_dim=len(movie_to_idx), output_dim=embedding_size)(movie)
    dot = Dot(name='dot_product', normalize=True, axes=2)(
        [link_embedding, movie_embedding])
    merged = Reshape((1,))(dot)
    model = Model(inputs=[link, movie], outputs=[merged])
    model.compile(optimizer='nadam', loss='mse')
    return model

model = movie_embedding_model()
```

我们将使用生成器来喂养模型。生成器产生由正样本和负样本组成的数据批次。

我们从对数组中的正样本进行采样，然后用负样本填充它。负样本是随机选择的，并确保它们不在`pairs_set`中。然后我们以我们的网络期望的格式返回数据，即输入/输出元组：

```py
def batchifier(pairs, positive_samples=50, negative_ratio=5):
    batch_size = positive_samples * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    while True:
        for idx, (link_id, movie_id) in enumerate(
                random.sample(pairs, positive_samples)):
            batch[idx, :] = (link_id, movie_id, 1)
        idx = positive_samples
        while idx < batch_size:
            movie_id = random.randrange(len(movie_to_idx))
            link_id = random.randrange(len(top_links))
            if not (link_id, movie_id) in pairs_set:
                batch[idx, :] = (link_id, movie_id, -1)
                idx += 1
        np.random.shuffle(batch)
        yield {'link': batch[:, 0], 'movie': batch[:, 1]}, batch[:, 2]
```

训练模型的时间：

```py
positive_samples_per_batch=512

model.fit_generator(
    batchifier(pairs,
               positive_samples=positive_samples_per_batch,
               negative_ratio=10),
    epochs=25,
    steps_per_epoch=len(pairs) // positive_samples_per_batch,
    verbose=2
)
```

训练时间将取决于您的硬件，但如果您从10,000部电影数据集开始，即使在没有GPU加速的笔记本电脑上，训练时间也应该相当短。

我们现在可以通过访问`movie_embedding`层的权重从我们的模型中提取电影嵌入。我们对它们进行归一化，以便我们可以使用点积作为余弦相似度的近似：

```py
movie = model.get_layer('movie_embedding')
movie_weights = movie.get_weights()[0]
lens = np.linalg.norm(movie_weights, axis=1)
normalized = (movie_weights.T / lens).T
```

现在让我们看看嵌入是否有些意义：

```py
def neighbors(movie):
    dists = np.dot(normalized, normalized[movie_to_idx[movie]])
    closest = np.argsort(dists)[-10:]
    for c in reversed(closest):
        print(c, movies[c][0], dists[c])

neighbors('Rogue One')
```

```py
29 Rogue One 0.9999999
3349 Star Wars: The Force Awakens 0.9722805
101 Prometheus (2012 film) 0.9653338
140 Star Trek Into Darkness 0.9635347
22 Jurassic World 0.962336
25 Star Wars sequel trilogy 0.95218825
659 Rise of the Planet of the Apes 0.9516557
62 Fantastic Beasts and Where to Find Them (film) 0.94662267
42 The Avengers (2012 film) 0.94634
37 Avatar (2009 film) 0.9460137
```

## 讨论

嵌入是一种有用的技术，不仅适用于单词。在这个示例中，我们训练了一个简单的网络，并为电影生成了嵌入，取得了合理的结果。这种技术可以应用于任何我们有办法连接项目的时间。在这种情况下，我们使用了维基百科的外链，但我们也可以使用内链或页面上出现的单词。

我们在这里训练的模型非常简单。我们只需要让它提供一个嵌入空间，使得电影的向量和链接的向量的组合可以用来预测它们是否会共同出现。这迫使网络将电影投影到一个空间中，使得相似的电影最终位于相似的位置。我们可以使用这个空间来找到相似的电影。

在Word2vec模型中，我们使用一个词的上下文来预测这个词。在这个示例中，我们不使用链接的上下文。对于外部链接来说，这似乎不是一个特别有用的信号，但如果我们使用的是内部链接，这可能是有意义的。链接到电影的页面以一定的顺序进行链接，我们可以利用链接的上下文来改进我们的嵌入。

或者，我们可以使用实际的Word2vec代码，并在链接到电影的任何页面上运行它，但保留电影链接作为特殊标记。这样就会创建一个混合的电影和单词嵌入空间。

# 4.3 建立电影推荐系统

## 问题

如何基于嵌入构建推荐系统？

## 解决方案

使用支持向量机将排名靠前的项目与排名靠后的项目分开。

前面的方法让我们对电影进行聚类，并提出建议，比如“如果你喜欢《侠盗一号》，你也应该看看《星际穿越》。”在典型的推荐系统中，我们希望根据用户评分的一系列电影来显示建议。就像我们在[第3章](ch03.html#word_embeddings)中所做的那样，我们可以使用SVM来做到这一点。让我们按照*滚石*杂志2015年评选的最佳和最差电影，并假装它们是用户评分：

```py
best = ['Star Wars: The Force Awakens', 'The Martian (film)',
        'Tangerine (film)', 'Straight Outta Compton (film)',
        'Brooklyn (film)', 'Carol (film)', 'Spotlight (film)']
worst = ['American Ultra', 'The Cobbler (2014 film)',
         'Entourage (film)', 'Fantastic Four (2015 film)',
         'Get Hard', 'Hot Pursuit (2015 film)', 'Mortdecai (film)',
         'Serena (2014 film)', 'Vacation (2015 film)']
y = np.asarray([1 for _ in best] + [0 for _ in worst])
X = np.asarray([normalized_movies[movie_to_idx[movie]]
                for movie in best + worst])
```

基于此构建和训练一个简单的SVM分类器很容易：

```py
clf = svm.SVC(kernel='linear')
clf.fit(X, y)
```

我们现在可以运行新的分类器，打印出数据集中所有电影中最好的五部和最差的五部：

```py
estimated_movie_ratings = clf.decision_function(normalized_movies)
best = np.argsort(estimated_movie_ratings)
print('best:')
for c in reversed(best[-5:]):
    print(c, movies[c][0], estimated_movie_ratings[c])

print('worst:')
for c in best[:5]:
    print(c, movies[c][0], estimated_movie_ratings[c])
```

```py
best:
(6870, u'Goodbye to Language', 1.24075226186855)
(6048, u'The Apu Trilogy', 1.2011876298842317)
(481, u'The Devil Wears Prada (film)', 1.1759994747169913)
(307, u'Les Mis\xe9rables (2012 film)', 1.1646775074857494)
(2106, u'A Separation', 1.1483743944891462)
worst:
(7889, u'The Comebacks', -1.5175929012505527)
(8837, u'The Santa Clause (film series)', -1.4651252650867073)
(2518, u'The Hot Chick', -1.464982008376793)
(6285, u'Employee of the Month (2006 film)', -1.4620595013243951)
(7339, u'Club Dread', -1.4593221506016203)
```

## 讨论

正如我们在上一章中看到的，我们可以使用支持向量机高效地构建一个区分两个类别的分类器。在这种情况下，我们让它根据我们先前学习到的嵌入来区分好电影和坏电影。

由于支持向量机找到一个或多个超平面来将“好”的示例与“坏”的示例分开，我们可以将其用作个性化功能——距离分隔超平面最远且在右侧的电影应该是最受喜爱的电影。

# 4.4 预测简单的电影属性

## 问题

您想要预测简单的电影属性，比如烂番茄评分。

## 解决方案

使用学习到的嵌入模型的向量进行线性回归模型，以预测电影属性。

让我们尝试一下这个方法来处理烂番茄评分。幸运的是，它们已经以`movie[-2]`的形式作为字符串存在于我们的数据中，形式为`*N*%`：

```py
rotten_y = np.asarray([float(movie[-2][:-1]) / 100
                       for movie in movies if movie[-2]])
rotten_X = np.asarray([normalized_movies[movie_to_idx[movie[0]]]
                       for movie in movies if movie[-2]])
```

这应该为我们大约一半的电影提供数据。让我们在前80%的数据上进行训练：

```py
TRAINING_CUT_OFF = int(len(rotten_X) * 0.8)
regr = LinearRegression()
regr.fit(rotten_X[:TRAINING_CUT_OFF], rotten_y[:TRAINING_CUT_OFF])
```

现在让我们看看我们在最后20%的进展如何：

```py
error = (regr.predict(rotten_X[TRAINING_CUT_OFF:]) -
         rotten_y[TRAINING_CUT_OFF:])
'mean square error %2.2f' % np.mean(error ** 2)
```

```py
mean square error 0.06
```

看起来真的很令人印象深刻！但虽然这证明了线性回归的有效性，但我们的数据存在一个问题，使得预测烂番茄评分变得更容易：我们一直在训练前10000部电影，而热门电影并不总是更好，但平均来说它们得到更高的评分。

通过将我们的预测与始终预测平均分数进行比较，我们可以大致了解我们的表现如何：

```py
error = (np.mean(rotten_y[:TRAINING_CUT_OFF]) - rotten_y[TRAINING_CUT_OFF:])
'mean square error %2.2f' % np.mean(error ** 2)
```

```py
'mean square error 0.09'
```

我们的模型确实表现得更好一些，但基础数据使得产生一个合理的结果变得容易。

## 讨论

复杂的问题通常需要复杂的解决方案，深度学习肯定可以给我们这些。然而，从可能起作用的最简单的事物开始通常是一个不错的方法。这让我们能够快速开始，并让我们知道我们是否朝着正确的方向努力：如果简单模型根本不产生任何有用的结果，那么复杂模型帮助的可能性不大，而如果简单模型有效，更复杂的模型有很大机会帮助我们取得更好的结果。

线性回归模型就是最简单的模型之一。该模型试图找到一组因素，使这些因素与我们的向量的线性组合尽可能接近目标值。与大多数机器学习模型相比，这些模型的一个好处是我们实际上可以看到每个因素的贡献是什么。
