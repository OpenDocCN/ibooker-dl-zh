# 附录 B. K-最近邻和支持向量机

在本附录中，我们考察了具有更多计算性质的经典机器学习算法，这些算法在书中没有涉及，因为它们现在使用较少，并且与大多数应用中的决策树集成相比被认为是过时的。总的来说，支持向量机（SVMs）仍然是一种适合于高维、噪声或小规模数据应用的实用机器学习算法。另一方面，k-最近邻（k-NN）非常适合在数据特征较少、可能存在异常值且预测不需要高度准确性的应用中运行。例如，SVMs 仍然可以用于分类医学图像，如乳腺 X 光片和 X 射线；在汽车行业中用于车辆检测和跟踪；或用于检测电子邮件垃圾邮件。相反，k-NN 主要应用于推荐系统，特别是基于用户过去行为的协同过滤方法，以推荐产品或服务。

它们在大多数表格数据情况下都适用，当您的数据不是太小或太大时——作为一个经验法则，当行数少于 10,000 行时。我们将从 k-NN 算法开始，这是数据科学家在机器学习问题中使用了几十年的算法，它易于理解和实现。然后我们将通过 SVMs 和关于使用 GPU 以在中等规模数据集上运行这些算法的简要说明来完成我们的概述。所有示例都需要第四章中介绍的 Airbnb 纽约市数据集。您可以通过执行以下代码片段来重新执行它：

```py
import numpy as np
import pandas as pd
excluding_list = [
    'price', 'id', 'latitude', 'longitude', 
    'host_id', 'last_review', 'name', 'host_name'
]                                                ①
categorical = [
    'neighbourhood_group', 'neighbourhood', 
    'room_type'
]                                                ②
continuous = [
    'minimum_nights', 'number_of_reviews', 'reviews_per_month', 
    'Calculated_host_listings_count'
]                                                ③
data = pd.read_csv("./AB_NYC_2019.csv")
target_median = (
    data["price"] > data["price"].median()
).astype(int)                                    ④
```

① 要排除的分析列名列表

② 可能代表数据集中分类变量的列名列表

③ 代表数据集中连续数值变量的列名列表

④ 二元平衡目标

代码将加载您的数据集并定义要排除的分析特征或考虑为连续或分类变量以进行处理的特征。

## B.1 k-NN

k-NN 算法适用于回归和分类任务，被认为是制作预测的最简单和最直观的算法之一。它从训练集中找到 k 个（k 是一个整数）最近的例子，并使用它们的信息进行预测。例如，如果任务是回归，它将取 k 个最近例子平均值。如果任务是分类，它将在 k 个最近例子中选择最常见的类别。

从技术上讲，k-NN 通常被认为是一种基于实例的学习算法，因为它以原样记住训练示例。它也被认为是一种“懒惰算法”，因为与大多数机器学习算法相反，在训练时间几乎没有处理。在训练期间，通常会有一些通过优化算法和数据结构处理距离的过程，这使得在训练示例附近查找邻近点之后的计算成本较低。大部分的计算工作是在测试时间完成的（见图 B.1）。

![图片](img/APPB_F01_Ryan2.png)

图 B.1 使用 k = 3 的 k-NN 对新样本（三角形）进行分类

我们将 k-NN 分类器应用于第四章中提到的 Airbnb 纽约市数据，如附录 B.1 所示。由于 k-NN 是基于距离工作的，为了获得一个有效的解决方案，特征必须在同一尺度上，从而确保在距离测量过程中每个维度都有相同的权重。如果一个特征在不同的或较小的尺度上，它会在过程中被过度加权。如果较大的尺度描述了一个特征，则相反的情况会发生。为了说明这个问题，让我们考虑当我们基于千米、米和厘米比较距离时会发生什么。即使距离是可比较的，米和厘米的数值将超过千米测量值。这个问题通常通过缩放特征来解决——例如，通过减去它们的平均值并除以它们的标准差（这种操作称为 z 分数标准化）。

或标准化）。此外，诸如降维或特征选择等技术对于此算法也是有帮助的，因为重新排列预测因子或不同的预测因子集可能会导致在问题上的预测性能有所提高或降低。

在我们的案例中，正如表格数据通常所做的那样，情况因分类特征而复杂化，这些特征一旦进行独热编码，就会变成从 0 到 1 的二进制值，其比例与归一化特征不同。我们提出的解决方案是首先对数值特征进行离散化，从而有效地将它们转换为二进制特征，每个特征表示一个特征的数值是否将落在特定的范围内。连续特征的二值化是通过嵌入到`numeric_discretizing`管道中的 KBinsDiscretizer 类（[`mng.bz/N12N`](https://mng.bz/N12N)）实现的，它将每个数值特征转换为五个二进制特征，每个特征覆盖一个值范围。在处理时间，我们还应用主成分分析（PCA）来降低维度并使所有特征无关。然而，我们可能会减弱数据中的非线性，因为 PCA 是一种基于变量线性组合的技术。PCA 处理的数据具有不相关结果特征的特点，这适合 k-NN：k-NN 基于距离，如果维度无关，距离测量才能正确工作。因此，任何距离变化都是由于单个维度的变化，而不是多个维度的变化。以下列表显示了实现数据转换过程和训练 k-NN 的代码。

列表 B.1 k-NN 分类器

```py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score

categorical_onehot_encoding = OneHotEncoder(handle_unknown='ignore')

accuracy = make_scorer(accuracy_score)                            ①
cv = KFold(5, shuffle=True, random_state=0)                       ②
model = KNeighborsClassifier(n_neighbors=30,
                             weights="uniform",
                             algorithm="auto",
                             n_jobs=-1)                           ③

column_transform = ColumnTransformer(
    [('categories', categorical_onehot_encoding, low_card_categorical),
     ('numeric', numeric_discretizing, continuous)],
    remainder='drop',
    verbose_feature_names_out=False,
    sparse_threshold=0.0)                                         ④

model_pipeline = Pipeline(
    [('processing', column_transform),
     ('pca', PCA(n_components="mle")),
     ('modeling', model)])                                        ⑤

cv_scores = cross_validate(estimator=model_pipeline,
                           X=data,
                           y=target_median,
                           scoring=accuracy,
                           cv=cv,
                           return_train_score=True,
                           return_estimator=True)                 ⑥

mean_cv = np.mean(cv_scores['test_score'])
std_cv = np.std(cv_scores['test_score'])
fit_time = np.mean(cv_scores['fit_time'])
score_time = np.mean(cv_scores['score_time'])
print(f"{mean_cv:0.3f} ({std_cv:0.3f})",
      f"fit: {fit_time:0.2f}",
      f"secs pred: {score_time:0.2f} secs")                       ⑦
```

① 使用 accuracy_score 度量创建一个评分函数

② 创建一个具有洗牌和固定随机状态的五折交叉验证迭代器

③ 创建一个具有指定超参数的 KNeighborsClassifier 实例

④ 定义一个 ColumnTransformer 来预处理特征，对低基数分类特征应用独热编码，对数值特征应用离散化

⑤ 创建一个管道，按顺序应用列转换，执行 PCA 降维，然后将 k-nn 模型拟合到数据上

⑥ 使用定义的管道对数据进行交叉验证，使用准确率评分

⑦ 打印测试分数的均值和标准差

运行脚本后，您将获得一个接近朴素贝叶斯解决方案性能的结果：

```py
0.814 (0.005) fit: 0.13 secs pred: 8.75 secs
```

性能良好，尽管推理时间相对较高。由于此算法通过类比工作（它将在训练中寻找类似案例以获得可能的预测想法），因此在大数据集上表现更好，在大数据集中找到与要预测的实例相似的实例的可能性更高。自然地，数据集的正确大小由使用的特征数量决定，因为特征越多，算法需要更多的案例来很好地泛化。

虽然通常人们将重点放在设置 k 参数的最佳值，将其视为平衡算法对训练数据欠拟合和过拟合的关键，但我们反而将注意力转向其他方面，以有效地使用此模型。由于算法通过类比和复杂空间中的距离来工作，我们考虑了关于此方法两个重要的问题：

+   要测量的维度和维度诅咒

+   适当的距离度量以及如何处理特征

在 k-NN 中，分类或回归估计取决于基于特征计算的距离度量的最相似示例。然而，在数据集中，并非所有特征都可以被认为在判断一个示例与其他示例相似时很重要，并且并非所有特征都可以以相同的方式进行比较。在应用 k-NN 时，对问题的先验知识非常重要，因为你必须只选择与你要解决的问题相关的特征。如果你为问题组装了过多的特征，你将依赖于过多的复杂空间来导航。图 B.1 展示了 k-NN 算法如何仅使用两个特征（在 x 和 y 维度上表示）工作，你可以直观地理解，如果某个区域有混合的类别或者没有训练示例靠近新实例，那么对新实例进行分类（图中的三角形）可能很困难。你必须依赖于更远的那些实例。

游戏中出现了维度诅咒，它指出随着特征数量的增加，你需要更多的示例来保持数据点之间有意义的距离。此外，维度诅咒还意味着必要示例的数量会随着特征数量的增加而指数级增长。对于 k-NN 算法来说，这意味着如果你提供了过多的特征，而示例数量不足，它将在空空间中工作。寻找邻居将变得令人畏惧。此外，如果你只是组装了相关和不相关的特征，风险是算法可能会将一些与你要预测的案例非常远的示例标记为邻居，并且选择可能基于对问题无用的特征。因此，如果你打算使用 k-NN，你应该非常小心地选择要使用的特征（如果你不知道使用哪些，你需要依赖特征选择）或者非常熟悉问题，以确定应该将什么放入算法中。简约对于 k-NN 的正确工作至关重要。

当你决定好特征后，关于你将使用的距离度量，你需要标准化、去除冗余和转换特征。这是因为距离度量基于绝对测量，不同的尺度可以以不同的方式权衡。考虑使用千米、米和厘米的测量值。厘米可能会占主导地位，因为它们很容易就有最大的数字。此外，具有相似特征（多重共线性问题）可能导致距离测量对某些特征集的权重超过其他特征集。最后，距离测量意味着具有相同的维度进行测量。然而，在数据集中，你可能会发现不同类型的数据——数值、分类和时间相关的数据，它们通常需要在距离计算中更好地结合在一起，因为它们具有不同的数值特征。

因此，除了事先仔细选择要使用的特征外，当使用 k-NN 时，我们建议使用所有同种类的特征（或所有数值或所有分类）来标准化它们，如果需要的话，并通过如 PCA（[`mng.bz/8OrZ`](https://mng.bz/8OrZ)）等方法减少它们的信噪比，这将重新制定数据集成为一个新的数据集，其中特征之间不相关。

## B.2 支持向量机（SVMs）

在 2010 年之前，SVMs 以表格问题中最有前途的算法而闻名。然而，在过去 10 年中，基于树的模型已经超越了 SVMs，成为表格数据的首选方法。然而，SVMs 仍然是一系列处理二元、多类、回归和异常/新颖性检测的技术。它们基于这样的想法：如果你的观察结果可以表示为多维空间中的点，那么存在一个超平面（即穿过多个维度的分离平面）可以将它们分开成类别或值，通过确保它们之间最大的分离，也保证了最稳健和可靠的预测。图 B.2 展示了一个简单的 SVM 应用到具有两个特征的二元分类问题示例，这些特征在 x 轴和 y 轴上表示，作为预测因子。SVM 模型产生了一条分隔线，在两组之间有最大的松弛空间，如图所示，其中虚线界定松弛空间。在这样做的时候，它只考虑靠近分隔器的几个点，称为支持向量。相反，它忽略了靠近但会混淆算法的点，例如，它们在错误的一侧。它还忽略了远离分隔线远离的点。异常值对这种算法的影响很小。

![图片](img/APPB_F02_Ryan2.png)

图 B.2 SVM 中的一个分离超平面

SVM 的强点在于它们对过拟合、数据中的噪声和异常值的鲁棒处理，以及它们如何成功处理包含众多多重共线性特征的集合。将不同的非线性方法应用于数据时，SVM 不需要我们为逻辑回归所看到的变换（如多项式展开）。然而，它们可以使用基于领域的特征工程，就像所有其他机器学习算法一样。

在弱点方面，SVM 优化复杂，并且仅适用于有限数量的示例。此外，它们最适合二元预测和仅用于类别预测；它们不是概率算法，你需要将它们与另一个算法（如逻辑回归）结合使用以进行校准（以从它们中提取概率）。这使得 SVM 在风险估计的有限范围内有效。

在我们的例子中，我们使用具有径向基函数核的二进制分类 SVM 和 Airbnb 纽约市数据重新应用我们的问题，这是一种能够自动建模提供特征之间复杂非线性关系的途径。

列表 B.2 支持向量机分类器

```py
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

numeric_standardization = Pipeline([
       ("imputation", SimpleImputer(strategy="constant", fill_value=0)),
       ("standardizing", StandardScaler())
       ])

accuracy = make_scorer(accuracy_score)                            ①
cv = KFold(5, shuffle=True, random_state=0)                       ②
model = SVC(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    probability=False
)                                                                 ③

column_transform = ColumnTransformer(
    [('categories', categorical_onehot_encoding, low_card_categorical),
     ('numeric', numeric_standardization, continuous)],
    remainder='drop',
    verbose_feature_names_out=False,
    sparse_threshold=0.0)                                         ④

model_pipeline = Pipeline(
    [('processing', column_transform),
     ('modeling', model)])                                        ⑤

cv_scores = cross_validate(estimator=model_pipeline,
                           X=data,
                           y=target_median,
                           scoring=accuracy,
                           cv=cv,
                           return_train_score=True,
                           return_estimator=True)                 ⑥

mean_cv = np.mean(cv_scores['test_score'])
std_cv = np.std(cv_scores['test_score'])
fit_time = np.mean(cv_scores['fit_time'])
score_time = np.mean(cv_scores['score_time'])
print(f"{mean_cv:0.3f} ({std_cv:0.3f})",
      f"fit: {fit_time:0.2f}",
      f"secs pred: {score_time:0.2f} secs")                       ⑦
```

① 使用 accuracy_score 度量创建一个评分函数

② 创建一个具有洗牌和固定随机状态的五折交叉验证迭代器

③ 创建一个具有指定超参数的支持向量机分类器实例

④ 定义一个 ColumnTransformer 以预处理特征，对低基数分类特征应用独热编码，对数值特征进行标准化

⑤ 创建一个管道，按顺序应用列转换

使用定义的管道在数据上执行交叉验证，使用准确率评分

⑦ 打印测试分数的均值和标准差

结果相当有趣，通过调整超参数可能还会变得更好：

```py
0.821 (0.004) fit: 102.28 secs pred: 9.80 secs
```

然而，训练单个折叠所需的时间与所有之前的机器学习算法相比过于冗长。在本附录的下一节中，我们将讨论如何使用 GPU 卡加速过程，同时仍然使用 Scikit-learn API。

## B.3 使用 GPU 进行机器学习

由于深度学习在数据科学领域的迅速崛起，GPU 现在在本地和云计算中都得到了广泛应用。以前，你只听说过 GPU 在 3D 游戏、图形处理渲染和动画中的应用。由于它们便宜且擅长快速矩阵乘法任务，学者和实践者迅速将 GPU 用于神经网络计算。RAPIDS 是由 NVIDIA（GPU 顶级制造商之一）开发的一系列用于在 GPU 上执行数据科学全光谱的包，而不仅仅是深度学习。RAPIDS 包承诺帮助机器学习管道的各个阶段，从端到端。这对许多经典机器学习算法来说是一个变革，特别是对于 SVMs，它是处理涉及噪声、异常值和大型数据集（特别是如果特征多线性或稀疏）的复杂任务的最可靠选择。在 RAPIDS 包（表 B.1）中，所有命令都采用了现有的 API 作为它们的命令。这确保了包的即时市场采用，对于用户来说，无需重新学习轮子的工作方式。

表 B.1 Rapids 包

| Rapids 包 | 任务 | API 模拟 |
| --- | --- | --- |
| cuPy | 数组操作 | NumPy |
| cuDF | 数据处理 | pandas |
| cuML | 机器学习 | Scikit-learn |

本节将重点介绍如何轻松地将你的 Scikit-learn 算法替换为 RAPIDS cuML 包。目前，此包包括线性模型、k-NN 和 SVMs 的实现，以及聚类和降维。以下列表显示了测试支持向量分类器（使用我们在上一节中尝试的径向基函数核）的 RAPIDS 实现（使用 P100 GPU）的代码。

列表 B.3 RAPIDS cuML 支持向量分类器

```py
from cuml.svm import SVC
from sklearn.metrics import accuracy_score

accuracy = make_scorer(accuracy_score)                            ①
cv = KFold(5, shuffle=True, random_state=0)                       ②
model = SVC(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    probability=False
)                                                                 ③

column_transform = ColumnTransformer(
    [('categories', categorical_onehot_encoding, low_card_categorical),
     ('numeric', numeric_standardization, continuous)],
    remainder='drop',
    verbose_feature_names_out=False,
    sparse_threshold=0.0)                                         ④

model_pipeline = Pipeline(
    [('processing', column_transform),
     ('modeling', model)])                                        ⑤

cv_scores = cross_validate(estimator=model_pipeline,
                           X=data,
                           y=target_median,
                           scoring=accuracy,
                           cv=cv,
                           return_train_score=True,
                           return_estimator=True)                 ⑥

mean_cv = np.mean(cv_scores['test_score'])
std_cv = np.std(cv_scores['test_score'])
fit_time = np.mean(cv_scores['fit_time'])
score_time = np.mean(cv_scores['score_time'])
print(f"{mean_cv:0.3f} ({std_cv:0.3f})",
      f"fit: {fit_time:0.2f}",

      f"secs pred: {score_time:0.2f} secs")                       ⑦
```

① 使用 accuracy_score 度量创建评分函数

② 创建一个具有洗牌和固定随机状态的五折交叉验证迭代器

③ 从 GPU 加速的 cuML 库中创建一个支持向量分类器实例，并指定超参数

④ 定义一个 ColumnTransformer 来预处理特征，对低基数分类特征应用独热编码，对数值特征应用标准化

⑤ 创建一个管道，按顺序将列转换和模型应用于数据

使用定义的管道在数据上执行交叉验证，并使用准确率评分

⑦ 打印测试分数的均值和标准差

我们获得的结果是

```py
0.821 (0.004) fit: 4.09 secs pred: 0.11 secs
```

如您所见，我们通过重用相同的代码但依赖 cuML 获得了相同的结果。然而，处理时间已从每个文件夹 102 秒降低到每个文件夹 4 秒。如果您计算时间节省，那将是 25 倍的速度提升。确切的表现效益取决于您使用的 GPU 型号；GPU 越强大，结果越快，因为这与 GPU 卡从 CPU 内存传输数据以及处理矩阵乘法的速度有关。

基于标准 GPU（公众可访问的通用 GPU）上的此类性能，我们最近看到了将表格数据与深度学习模型（如文本或图像）的大嵌入融合的应用。SVMs（支持向量机）与众多特征（但不超过示例中的数量）和稀疏值（许多零值）配合良好。在这种情况下，SVMs 可以轻松获得最先进的结果，超越当时其他更受欢迎的表格算法——即 XGBoost 和其他梯度提升实现，以及端到端深度学习解决方案，后者在没有足够案例提供时表现较弱。

拥有 GPU 并使您的代码适应使用 RAPIDS 算法，使得某些经典的表格机器学习算法再次具有竞争力，这通常基于机器学习中没有免费午餐的原则（关于没有免费午餐定理的更多细节可在[`www.no-free-lunch.org/`](http://www.no-free-lunch.org/)找到）。考虑到您的项目限制（例如，您可能无法在项目环境中获得某些资源），如果可行，永远不要排除在先验测试中将您的问题与所有可用算法进行比较。
