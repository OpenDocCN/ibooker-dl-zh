# 第十章。构建反向图像搜索服务

在前一章中，我们看到如何在我们自己的图像上使用预训练网络，首先通过在网络顶部运行分类器，然后在一个更复杂的示例中，我们只训练网络的部分来识别新的图像类别。在本章中，我们将使用类似的方法来构建一个反向图像搜索引擎，或者通过示例搜索。

我们将从查找维基数据查询维基百科获取一组良好的基础图像开始。然后，我们将使用预训练网络提取每个图像的值以获得嵌入。一旦我们有了这些嵌入，找到相似的图像只是最近邻搜索的一个简单问题。最后，我们将研究主成分分析（PCA）作为一种可视化图像之间关系的方法。

本章的代码可以在以下笔记本中找到：

```py
10.1 Building an inverse image search service
```

# 10.1 从维基百科获取图像

## 问题

如何从维基百科获取一组覆盖主要类别的干净图像？

## 解决方案

使用维基数据的元信息来查找代表一类事物的维基百科页面。

维基百科包含大量可用于大多数目的的图片。然而，其中绝大多数图片代表具体实例，这并不是反向搜索引擎所需要的。我们希望返回代表猫作为一个物种的图片，而不是像加菲猫这样的特定猫。

维基数据，维基百科的结构化表亲，基于形式为（主题，关系，对象）的三元组，并且有大量编码的谓词，部分基于维基百科。其中之一“实例”由`P31`表示。我们要找的是实例关系中对象的图像列表。我们可以使用维基数据查询语言来请求这个：

```py
query = """SELECT DISTINCT ?pic
WHERE
{
 ?item wdt:P31 ?class .
 ?class wdt:P18 ?pic
}
"""
```

我们可以使用请求调用维基数据的查询后端，并将结果 JSON 展开为图像引用列表：

```py
url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
data = requests.get(url, params={'query': query, 'format': 'json'}).json()
images = [x['pic']['value'] for x in data['results']['bindings']]
```

返回的引用是指向图像页面的 URL，而不是图像本身。各种维基项目中的图像应该存储在*http://upload.wikimedia.org/wikipedia/commons/*，但不幸的是，这并不总是情况—有些仍然存储在特定语言的文件夹中。因此，我们还必须至少检查英语文件夹（*en*）。图像的实际 URL 由文件名和文件名的 MD5 哈希的`hexdigest`的前两个字符确定。如果我们需要多次执行此操作，则将图像缓存到本地会有所帮助：

```py
def center_crop_resize(img, new_size):
    w, h = img.size
    s = min(w, h)
    y = (h - s) // 2
    x = (w - s) // 2
    img = img.crop((x, y, s, s))
    return img.resize((new_size, new_size))

def fetch_image(image_cache, image_url):
    image_name = image_url.rsplit('/', 1)[-1]
    local_name = image_name.rsplit('.', 1)[0] + '.jpg'
    local_path = os.path.join(image_cache, local_name)
    if os.path.isfile(local_path):
        img = Image.open(local_path)
        img.load()
        return center_crop_resize(img, 299)
    image_name = unquote(image_name).replace(' ', '_')
    m = md5()
    m.update(image_name.encode('utf8'))
    c = m.hexdigest()
    for prefix in ('http://upload.wikimedia.org/wikipedia/en',
                   'http://upload.wikimedia.org/wikipedia/commons'):
        url = '/'.join((prefix, c[0], c[0:2], image_name))
        r = requests.get(url)
        if r.status_code != 404:
            try:
                img = Image.open(BytesIO(r.content))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(local_path)
                return center_crop_resize(img, 299)
            except IOError:
                pass
    return None
```

即使这样有时候也不起作用。本章的笔记本包含一些更多处理边缘情况的代码，以增加图像的产出。

现在我们需要做的就是获取图像。这可能需要很长时间，因此我们使用`tqdm`来显示我们的进度：

```py
valid_images = []
for image_name in tqdm(images):
    img = fetch_image(IMAGE_DIR, image_name)
    if img:
        valid_images.append(img)
```

## 讨论

维基数据的查询语言并不广为人知，但它是访问结构化数据的有效方式。这里的示例非常简单，但在线上可以找到更复杂的查询，例如返回世界上最大的女市长或虚构角色中最流行的姓氏。这些数据大部分也可以从维基百科中提取，但运行维基数据查询通常更快、更精确且更有趣。

维基媒体宇宙也是图像的良好来源。有数千万张可用的图像，全部都有友好的重用许可证。此外，使用维基数据，我们可以访问这些图像的各种属性。很容易扩展这个方法，不仅返回图像 URL，还返回图像中对象的名称，可以选择使用我们选择的语言。

###### 注意

这里描述的`fetch_image`函数大多数时候都有效，但并非总是有效。我们可以通过获取从维基数据查询返回的 URL 的内容，并从 HTML 代码中提取`<img>`标签来改进这一点。

# 10.2 将图像投影到 N 维空间

## 问题

给定一组图像，如何组织它们以使相似的图像彼此靠近？

## 解决方案

将图像识别网络的倒数第二层的权重视为图像嵌入。这一层直接连接到绘制结论的`softmax`层。因此，网络认为是猫的任何东西应该具有相似的值。

让我们加载并实例化预训练网络。我们将再次使用 Inception——让我们使用`.summary()`来查看其结构：

```py
base_model = InceptionV3(weights='imagenet', include_top=True)
base_model.summary()
```

正如您所看到的，我们需要`avg_pool`层，其大小为 2,048：

```py
model = Model(inputs=base_model.input,
              outputs=base_model.get_layer('avg_pool').output)
```

现在我们可以在一张图像或一组图像上运行模型：

```py
def get_vector(img):
    if not type(img) == list:
        images = [img]
    else:
        images = img
    target_size = int(max(model.input.shape[1:]))
    images = [img.resize((target_size, target_size), Image.ANTIALIAS)
              for img in images]
    np_imgs = [image.img_to_array(img) for img in images]
    pre_processed = preprocess_input(np.asarray(np_imgs))
    return model.predict(pre_processed)
```

并对我们在上一个示例中获取的图像进行分块索引（当然，如果您有足够的内存，可以尝试一次性完成整个操作）：

```py
chunks = [get_vector(valid_images[i:i+256])
          for i in range(0, len(valid_images), 256)]
vectors = np.concatenate(chunks)
```

## 讨论

在这个示例中，我们使用了网络的倒数第二层来提取嵌入。由于这一层直接连接到确定实际输出层的`softmax`层，我们期望权重形成一个语义空间，将所有猫的图像大致放在同一空间中。但如果我们选择不同的层会发生什么呢？

将卷积网络视为图像识别的特征检测器，逐层增加抽象级别。最低级别直接处理像素值，并将检测到非常局部的模式。最后一层检测“猫”的概念。

选择一个较低的层应该会导致在较低的抽象级别上的相似性，所以我们不会返回类似猫的东西，而是期望看到具有相似纹理的图像。

# 10.3 在高维空间中找到最近邻

## 问题

如何在高维空间中找到彼此最接近的点？

## 解决方案

使用 scikit-learn 的*k*-最近邻实现。

*k*-最近邻算法构建一个模型，可以快速返回最近邻。虽然会有一些精度损失，但比进行精确计算要快得多。实际上，它在我们的向量上构建了一个距离索引：

```py
nbrs = NearestNeighbors(n_neighbors=10,
                        balgorithm='ball_tree').fit(vectors)
```

有了这个距离索引，我们现在可以快速返回给定输入图像的图像集中的近似匹配。我们已经实现了逆图像搜索！让我们把所有这些放在一起找更多的猫：

```py
cat = get_vector(Image.open('data/cat.jpg'))
distances, indices = nbrs.kneighbors(cat)
```

并使用内联 HTML 图像显示前几个结果：

```py
html = []
for idx, dist in zip(indices[0], distances[0]):
    b = BytesIO()
    valid_images[idx].save(b, format='jpeg')
    b64_img = base64.b64encode(b.getvalue()).decode('utf-8'))
    html.append("<img src='data:image/jpg;base64,{0}'/>".format(b64_img)
HTML(''.join(html))
```

您应该看到一个由猫主导的漂亮图像列表！

## 讨论

在机器学习中，快速计算最近邻是一个活跃的研究领域。最简单的邻居搜索实现涉及在数据集中所有点对之间计算距离，如果我们在高维空间中有大量点，这种方法很快就会失控。

Scikit-learn 为我们提供了许多预先计算树的算法，可以帮助我们快速找到最近邻居，但会消耗一些内存。不同的方法在[文档](http://scikit-learn.org/stable/modules/neighbors.html)中有所讨论，但一般的方法是使用算法递归地将空间分割成子空间，从而构建树。这样我们可以快速识别在寻找邻居时要检查哪些子空间。

# 10.4 在嵌入中探索局部邻域

## 问题

您想探索图像的局部聚类是什么样子。

## 解决方案

使用主成分分析找到在局部图像集中最能区分图像的维度。

例如，假设我们有与我们的猫图像最接近的 64 张图像：

```py
nbrs64 = NearestNeighbors(n_neighbors=64, algorithm='ball_tree').fit(vectors)
distances64, indices64 = nbrs64.kneighbors(cat)
```

PCA 允许我们以尽可能少的损失降低空间的维度。如果我们将维度降低到二维，PCA 将找到可以将提供的示例投影到的平面，以尽可能少的损失。然后，我们看看这些示例在平面上的位置，就可以很好地了解本地邻域的结构。在这种情况下，我们将使用`TruncatedSVD`实现：

```py
vectors64 = np.asarray([vectors[idx] for idx in indices64[0]])
svd = TruncatedSVD(n_components=2)
vectors64_transformed = svd.fit_transform(vectors64)
```

`vectors64_transformed`现在的形状是 64×2。我们将在一个 8×8 的网格上绘制这 64 个图像，每个单元格的大小为 75×75。让我们首先将坐标归一化到 0 到 1 的比例上：

```py
mins = np.min(vectors64_transformed, axis=0)
maxs = np.max(vectors64_transformed, axis=0)
xys = (vectors64_transformed - mins) / (maxs - mins)
```

现在我们可以绘制并显示本地区域：

```py
img64 = Image.new('RGB', (8 * 75, 8 * 75), (180, 180, 180))

for idx, (x, y) in zip(indices64[0], xys):
    x = int(x * 7) * 75
    y = int(y * 7) * 75
    img64.paste(valid_images[idx].resize((75, 75)), (x, y))

img64
```

![本地聚类的图像](img/dlcb_10in01.png)

我们看到一个猫图像大致位于中间，一个角被动物主导，其余的图像由于其他原因匹配。请注意，我们在现有图像上绘制，因此网格实际上不会完全填满。

## 讨论

在食谱 3.3 中，我们使用 t-SNE 将高维空间折叠成二维平面。在这个食谱中，我们改用主成分分析。这两种算法实现了相同的目标，即降低空间的维度，但是它们的方式不同。

t-SNE 试图保持空间中点之间的距离相同，尽管降低了维度。在这种转换中当然会丢失一些信息，因此我们可以选择是尝试保持簇在本地保持完整（在高维度中彼此接近的点之间的距离保持相似）还是保持簇之间的距离保持完整（在高维度中彼此远离的点之间的距离保持相似）。

PCA 试图找到一个* N *维的超平面，该超平面与空间中的所有项目尽可能接近。如果* N *为 2，我们谈论的是一个普通平面，因此它试图找到在我们的高维空间中与所有图像最接近的平面。换句话说，它捕捉了两个最重要的维度（主成分），然后我们用它们来可视化猫空间。
