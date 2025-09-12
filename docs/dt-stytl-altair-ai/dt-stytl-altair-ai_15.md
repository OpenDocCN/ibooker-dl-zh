# 附录 B Python pandas DataFrame

本附录描述了 pandas DataFrame 的概述以及本书中使用的方法。

## B.1 pandas DataFrame 概述

Python pandas 是一个数据操作、分析和可视化库。它提供了加载数据、允许您操作、分析和可视化数据的工具。在本书中，我们使用 pandas DataFrame，它由行和列组成的二维结构。DataFrame 以表格形式存储数据，使您能够快速轻松地操作、分析、过滤和汇总数据。

创建 pandas DataFrame 有不同的方法。在本书中，我们考虑两种方法：从 Python 字典和从 CSV 文件。您可以从本书的 GitHub 存储库中下载附录 B/Pandas DataFrame.ipynb 中描述的代码。

### B.1.1 从字典构建

考虑以下列表，它从 Python 字典创建一个 pandas DataFrame。

##### 列表 B.1 从字典创建 DataFrame

```py
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'BirthDate': ['2000-01-30', '2001-02-03', '2001-04-05'],
    'MathsScore': [90, 85, None],
    'PhysicsScore': [87, 92, 89],
    'ChemistryScore': [92, None, 90],
    'Grade' : ['A', 'B', 'A']
}                                  #1

df = pd.DataFrame(data)    #2

df['BirthDate'] = pd.to_datetime(df['BirthDate'], format='%Y-%m-%d')    #3
```

#1 定义字典

#2 创建 DataFrame

#3 将 BirthDate 字段解析为日期

注意：使用`DataFrame()`从字典创建一个新的 DataFrame。

### B.1.2 从 CSV 文件构建

使用`read_csv()`文件将 CSV 文件加载到 pandas DataFrame 中。

##### 列表 B.2 从 CSV 文件创建 DataFrame

```py
import pandas as pd

df = pd.read_csv('data.csv')
```

注意：使用`read_csv()`从 CSV 文件创建一个新的 DataFrame。

现在您已经看到了如何创建 pandas DataFrame，我们将讨论本书中使用的主要 DataFrame 函数。

## B.2 dt

pandas DataFrame 中的`dt`变量允许您访问 Python 的内置 DateTime 库。使用它来存储和操作 DateTime 值，如年、月、日、小时、分钟和秒。考虑以下列表，它从 DateTime 列中提取年份。

##### 列表 B.3 如何使用 pandas `dt`

```py
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'BirthDate': ['2000-01-30', '2001-02-03', '2001-04-05'],
    'MathsScore': [90, 85, None],
    'PhysicsScore': [87, 92, 89],
    'ChemistryScore': [92, None, 90],
    'Grade' : ['A', 'B', 'A']
}

df = pd.DataFrame(data)
df['BirthDate'] = pd.to_datetime(df['BirthDate'], format='%Y-%m-%d')

year = df['BirthDate'].dt.year    #1
month = df['BirthDate'].dt.month     #2
day = df['BirthDate'].dt.day                       #3
weekOfYear = df['BirthDate'].dt.isocalendar().week   #4
```

#1 从 BirthDate 列提取年

#2 从 BirthDate 列提取月

#3 从 BirthDate 列提取天

#4 从 BirthDate 列提取周

注意：使用 pandas `dt`访问 Python DateTime 库的 DateTime 函数。

## B.3 groupby()

pandas `groupby()`方法根据某些列的值将数据拆分为组。此过程通常涉及为每个组创建一个汇总统计量，例如总和或平均值。

##### 列表 B.4 如何使用 pandas `groupby`

```py
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'BirthDate': ['2000-01-30', '2001-02-03', '2001-04-05'],
    'MathsScore': [90, 85, None],
    'PhysicsScore': [87, 92, 89],
    'ChemistryScore': [92, None, 90],
    'Grade' : ['A', 'B', 'A']
}

df = pd.DataFrame(data)
df_grouped = df.groupby(by='Grade').mean().reset_index()
```

注意：使用 pandas `groupby`按仪器分组，并按年级计算平均分数。使用`reset_index()`方法恢复索引列（例如示例中的`Grade`）。

表 B.1 显示了结果。

##### 表 B.1 列表 B.4 中`groupby()`的结果

| **成绩** | **数学成绩** | **物理成绩** | **化学成绩** |
| --- | --- | --- | --- |
| A  | 90.0  | 88.0  | 91.0  |
| B  | 85.0  | 92.0  |  |

## B.4 isnull()

pandas DataFrame 的`isnull()`方法返回一个新的布尔 DataFrame，指示 DataFrame 中的哪些值是 null（`NaN`）。使用此方法检测 DataFrame 中的缺失值。

##### 列表 B.5 如何使用 pandas `isnull`

```py
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'BirthDate': ['2000-01-30', '2001-02-03', '2001-04-05'],
    'MathsScore': [90, 85, None],
    'PhysicsScore': [87, 92, 89],
    'ChemistryScore': [92, None, 90],
    'Grade' : ['A', 'B', 'A']
}

df = pd.DataFrame(data)
df_isnull = df.isnull()
```

注意  使用 pandas 的 `isnull()` 函数检查 DataFrame 是否包含缺失值。您还可以将 `isnull()` 方法应用于单个列（例如，`df['ChemistryScore'].isnull()`）。

表 B.2 显示了结果，一个布尔 DataFrame，指示 DataFrame 中的哪些值是 null (`NaN`)。

##### 表 B.2 列表 B.5 中 `isnull()` 的结果

| **姓名** | **出生日期** | **MathsScore** | **PhysicsScore** | **ChemistryScore** | **Grade** |
| --- | --- | --- | --- | --- | --- |
| False | False | False | False | False | False |
| False | False | False | False | True | False |
| False | False | True | False | False | False |

## B.5 melt()

我们使用 pandas 的 `melt()` 函数通过将列转换为行来重塑数据。此函数将 DataFrame 从宽格式转换为长格式，可选地保留标识符。

##### 列表 B.6 如何使用 pandas `melt`

```py
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'BirthDate': ['2000-01-30', '2001-02-03', '2001-04-05'],
    'MathsScore': [90, 85, None],
    'PhysicsScore': [87, 92, 89],
    'ChemistryScore': [92, None, 90],
    'Grade' : ['A', 'B', 'A']
}

df = pd.DataFrame(data)

df_melted = df.melt(id_vars='Name', 
        var_name='Subject', 
        value_name='Score', 
        value_vars=['MathsScore', 'PhysicsScore', 'ChemistryScore']
)
```

注意  使用 pandas 的 `melt()` 函数将 DataFrame 从宽格式转换为长格式。将 `id_vars` 参数设置为指定要保留作为标识符的变量，并将 `var_name` 和 `value_name` 参数设置为结果熔化 DataFrame 中新变量的列名。使用 `value_vars` 选择要分组的列。

表 B.3 显示了列表 B.6 中使用的数据的熔化结果。

##### 表 B.3 列表 B.6 中描述的熔化操作的结果

| **姓名** | **科目** | **分数** |
| --- | --- | --- |
| 爱丽丝 | `MathsScore` | 90.0 |
| 鲍勃 | `MathsScore` | 85.0 |
| 查理 | `MathsScore` | Null |
| 爱丽丝 | `PhysicsScore` | 87.0 |
| 鲍勃 | `PhysicsScore` | 92.0 |
| 查理 | `PhysicsScore` | 89.0 |
| 爱丽丝 | `ChemistryScore` | 92.0 |
| 鲍勃 | `ChemistryScore` | Null |

## B.6 unique()

我们使用 pandas 的 `unique()` 方法从 DataFrame 的特定列中获取唯一值。此方法返回一个类似数组的对象，包含在指定列中找到的唯一值。

##### 列表 B.7 如何使用 pandas `unique`

```py
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'BirthDate': ['2000-01-30', '2001-02-03', '2001-04-05'],
    'MathsScore': [90, 85, None],
    'PhysicsScore': [87, 92, 89],
    'ChemistryScore': [92, None, 90],
    'Grade' : ['A', 'B', 'A']
}
df = pd.DataFrame(data)
unique_grades = df['Grade'].unique()
```

注意  使用 pandas 的 `unique()` 函数获取列的唯一值。

以下列表显示了计算 `Grade` 列（来自列表 B.7）的唯一值的结果。

##### 列表 B.8 pandas `unique()` 的结果

```py
array(['A', 'B'], dtype=object)
```

注意  此方法返回一个包含列唯一值的数组。
