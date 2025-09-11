# 附录：R 用户的 Python 入门指南

你可能会想要阅读和理解一些 Python 代码，甚至将一些 Python 代码转换成 R。本指南旨在使您能够尽快完成这些任务。正如您将看到的那样，R 和 Python 是足够相似的，以至于可以在不必学习所有 Python 的情况下完成这些任务。我们从容器类型的基础知识开始，逐步深入到类、双下划线、迭代器协议、上下文协议等机制！

## A.1 空白

在 Python 中，空白很重要。在 R 中，表达式通过 {} 分组成一个代码块。在 Python 中，通过使表达式共享缩进级别来完成。例如，具有 R 代码块的表达式可能是：

if (TRUE) {

cat("This is one expression. \n")

cat("This is another expression. \n")

}

Python 中的等价物：

if True:

print("This is one expression.")

print("This is another expression.")

Python 接受制表符或空格作为缩进间隔符，但当它们混合使用时，规则变得棘手。大多数样式指南建议（和 IDE 默认使用）只使用空格。

## A.2 容器类型

在 R 中，list() 是一个您可以使用来组织 R 对象的容器。R 的 list() 功能齐全，没有一个单一的直接等价物在 Python 中支持所有相同的功能。相反，您需要了解 (至少) 四种不同的 Python 容器类型：列表、字典、元组和集合。

### A.2.1 列表

Python 列表通常使用裸括号创建：[]。 （Python 内置的 list() 函数更像是一个强制转换函数，与 R 的 as.list() 的精神更接近）。关于 Python 列表最重要的一点是它们在原地修改。请注意在下面的示例中，y 反映了对 x 所做的更改，因为两个符号指向的底层列表对象是在原地修改的：

x = [1, 2, 3]

y = x➊

x.append(4)

print("x is", x)

x is [1, 2, 3, 4]

print("y is", y)

y is [1, 2, 3, 4]

➊ **现在 y 和 x 指向同一个列表！**

R 用户可能关注的一个 Python 成语是通过 append() 方法增长列表。在 R 中增长列表通常很慢，最好避免。但是因为 Python 的列表在原地修改（并且在添加项时避免了列表的完全复制），所以在原地增长 Python 列表是有效的。

在 Python 列表周围的一些语法糖可能会遇到的情况是 + 和 * 的使用。这些是连接和复制运算符，类似于 R 的 c() 和 rep()：

x = [1]

x

[1]

x + x

[1, 1]

x * 3

[1, 1, 1]

你可以使用尾随的 [] 来对列表进行索引，但请注意，索引是从 0 开始的：

x = [1, 2, 3]

x[0]

1

x[1]

2

x[2]

3

try:

x[3]

except Exception as e:

print(e)

列表索引超出范围

在索引时，负数从容器的末尾开始计数：

x = [1, 2, 3]

x[-1]

3

x[-2]

2

x[-3]

1

你可以在括号内使用冒号 (:) 对列表进行切片范围。请注意，切片语法*不包含*切片范围的结尾。您还可以选择指定步长：

x = [1, 2, 3, 4, 5, 6]

x[0:2]➊

[1, 2]

x[1:]➋

[2, 3, 4, 5, 6]

x[:-2]➌

[1, 2, 3, 4]

x[:]➍

[1, 2, 3, 4, 5, 6]

x[::2]➎

[1, 3, 5]

x[1::2]➏

[2, 4, 6]

➊ **获取索引位置为 0 和 1 的项，而不是 2。**

➋ **获取索引位置为 1 到结尾的项。**

➌ **获取从开头到倒数第二个的项。**

➍ **获取所有项（这种习惯用法用于复制列表，以防止原地修改）。**

➎ **获取所有项，步长为 2。**

➏ **获取从索引 1 到结尾的所有项，步长为 2。**

### A.2.2 元组

元组的行为类似于列表，除了它们不可变，而且它们没有像 append() 这样的原地修改方法。它们通常使用裸 () 构造，但括号并不严格要求，你可能会看到一个隐式元组只是由逗号分隔的一系列表达式定义。因为括号也可以用于指定类似于 (x + 3) * 4 这样的表达式中的运算顺序，所以需要一种特殊的语法来定义长度为 1 的元组：尾随逗号。元组最常见的用法是在接受可变数量参数的函数中遇到：

x = (1, 2)

type(x)➊

<class 'tuple'>

len(x)

2

x

(1, 2)

x = (1,)➋

type(x)

<class 'tuple'>

len(x)

1

x

(1,)

x = ()➌

print(f"{type(x) = }; {len(x) = }; {x = }")➍

type(x) = <class 'tuple'>; len(x) = 0; x = ()

x = 1, 2➎

type(x)

<class 'tuple'>

len(x)

2

x = 1,➏

type(x)

<class 'tuple'>

len(x)

1

➊ **长度为 2 的元组**

➋ **长度为 1 的元组**

➌ **长度为 0 的元组**

➍ **插值字符串文字的示例。你可以使用 glue::glue() 在 R 中进行字符串插值。**

➎ **同样是一个元组**

➏ **注意单个尾随逗号！这是一个元组！**

### 打包和解包

元组是 Python 中 *打包* 和 *解包* 语义的容器。Python 提供了在一个表达式中允许你赋值多个符号的便利。这被称为 *解包*。

例如：

x = (1, 2, 3)

a, b, c = x

a

1

b

2

c

3

你可以使用 zeallot::`%<-%` 从 R 中访问类似的解包行为。

元组解包可以发生在各种情境下，比如迭代：

xx = (("a", 1),

("b", 2))

for x1, x2 in xx:

print("x1 =", x1)

print("x2 =", x2)

x1 = a

x2 = 1

x1 = b

x2 = 2

如果你尝试将容器解包为错误数量的符号，Python 就会引发一个错误：

x = (1, 2, 3)

a, b, c = x➊

a, b = x➋

在 py_call_impl(callable, dots$args, dots$keywords) 中出错：

![图像](img/common01.jpg) ValueError: 太多值要解包（期望 2 个）

a, b, c, d = x➋

在 py_call_impl(callable, dots$args, dots$keywords) 中出错：

![图像](img/common01.jpg) ValueError: 没有足够的值来解包（期望 4 个，得到 3 个）

➊ **成功**

➋ **错误：x 的值太多，无法解包。**

➋ **错误：x 的值不足以解包。**

可以解包可变数量的参数，使用 * 作为符号的前缀（当我们谈论函数时，我们将再次看到 * 前缀）：

x = (1, 2, 3)

a, *the_rest = x

a

1

the_rest

[2, 3]

你还可以解包嵌套结构：

x = ((1, 2), (3, 4))

(a, b), (c, d) = x

### A.2.3 字典（Dictionaries）

字典（Dictionaries）与 R 的环境最相似。它们是一个容器，您可以通过名称检索项目，尽管在 Python 中名称（在 Python 的术语中称为*key*）不像在 R 中一样需要是字符串。它可以是具有 hash() 方法的任何 Python 对象（意味着它可以是几乎任何 Python 对象）。它们可以使用 `{key: value}` 这样的语法创建。与 Python 列表一样，它们是就地修改的。请注意，reticulate::r_to_py() 将 R 命名列表转换为字典：

d = {"key1": 1,

"key2": 2}

d2 = d

d

{'key1': 1, 'key2': 2}

d["key1"]

1

d["key3"] = 3

d2➊

{'key1': 1, 'key2': 2, 'key3': 3}

➊ **就地修改！**

与 R 的环境不同（而不像 R 的命名列表），您不能使用整数索引来从字典中获取特定索引位置的项。字典是*无序*容器（但是，从 Python 3.7 开始，字典会保留项目插入顺序）：

d = {"key1": 1, "key2": 2}

d[1]➊

在 py_call_impl(callable, dots$args, dots$keywords) 中出现错误：KeyError: 1

➊ **错误：整数 "1" 不是字典中的键之一。**

与 R 的命名列表语义最接近的容器是 OrderedDict ([`mng.bz/7y5m`](http://mng.bz/7y5m))，但在 Python 代码中相对不常见，因此我们不再进一步介绍它。

### A.2.4 集合（Sets）

集合（Sets）是一个容器，可以用来有效地跟踪唯一项或去重列表。它们使用 {val1, val2} 构造（类似于字典，但没有 :）。将它们视为只使用键的字典。集合有许多高效的成员操作方法，如 intersection()、issubset()、union() 等：

s = {1, 2, 3}

类型（type）

<class 'set'>

s

{1, 2, 3}

s.add(1)

s

{1, 2, 3}

## A.3 使用 for 进行迭代

Python 中的 for 语句可用于遍历任何类型的容器：

for x in [1, 2, 3]:

print(x)

1

2

3

相比之下，R 具有相对有限的可以传递给 for 的对象集合。Python 则提供了迭代器协议接口，这意味着作者可以定义自定义对象，其行为由 for 调用（我们将在讨论类时有一个定义自定义可迭代对象的示例）。您可能希望使用 reticulate 从 R 使用 Python 可迭代对象，因此将语法糖稍微撕开一点，以显示 for 语句在 Python 中的工作原理，以及如何手动遍历它，这将会很有帮助。

发生了两件事：首先，从提供的对象构造了一个迭代器。然后，新的迭代器对象将重复调用 next()，直到耗尽为止：

l = [1, 2, 3]

it = iter(l)➊

it

<list_iterator object at 0x7f5e30fbd190>

➊ **创建一个迭代器对象。**

调用 next() 来遍历迭代器，直到迭代器耗尽为止：

next(it)

1

next(it)

2

next(it)

3

next(it)

在 py_call_impl(callable, dots$args, dots$keywords) 中出现错误：StopIteration

在 R 中，您可以使用 reticulate 以相同的方式遍历迭代器：

library(reticulate)

l <- r_to_py(list(1, 2, 3))

it <- as_iterator(l)

iter_next(it)

1.0

iter_next(it)

2.0

iter_next(it)

3.0

iter_next(it, completed = "StopIteration")

[1] "StopIteration"

遍历字典首先需要理解你是在遍历键、值还是两者都在。字典有允许你指定的方法：

d = {"key1": 1, "key2": 2}

for key in d:

print(key)

key1

key2

for value in d.values():

print(value)

1

2

for key, value in d.items():

print(key, ":", value)

key1 : 1

key2 : 2

### A.3.1 Comprehensions

推导式是特殊的语法，允许你构建类似列表或字典的容器，同时在每个元素上执行一个小操作或单个表达式。你可以把它看作是 R 中 lapply 的特殊语法。例如：

x = [1, 2, 3]

l = [element + 100 for element in x]

l

[101, 102, 103]

d = {str(element) : element + 100}

for element in x}

d

{'1': 101, '2': 102, '3': 103}

➊ **从 x 构建的列表推导式，其中每个元素加 100**

➋ **从 x 构建的字典推导式，其中键是一个字符串。Python 的 str()类似于 R 的 as.character()。**

## A.4 使用 def 定义函数

Python 函数使用 def 语句定义。指定函数参数和默认参数值的语法与 R 非常相似：

def my_function(name = "World"):

print("Hello", name)

my_function()

Hello World

my_function("Friend")

你好，朋友

等效的 R 片段将是：

my_function <- function(name = "World") {

cat("Hello", name, "\n")

}

my_function()

Hello World

my_function("Friend")

你好，朋友

与 R 函数不同，函数中的最后一个值不会自动返回。Python 需要一个明确的 return 语句：

def fn():

1

print(fn())

None

def fn():

返回 1

print(fn())

1

> 注意 对于高级 R 用户，Python 没有 R 的参数“promises”的等价物。函数参数默认值在函数构造时只计算一次。如果你将一个可变对象作为默认参数值定义为 Python 函数，这可能会让人感到惊讶！

def my_func(x = []):

x.append("was called")

print(x)

my_func()

my_func()

my_func()

['was called']

['was called', 'was called']

['was called', 'was called', 'was called']

你也可以定义 Python 函数，它接受可变数量的参数，类似于 R 中的…

def my_func(*args, **kwargs):

print("args =", args)

print("kwargs =", kwargs)

my_func(1, 2, 3, a = 4, b = 5, c = 6)

args = (1, 2, 3)

kwargs = {'a': 4, 'b': 5, 'c': 6}

➊ **args 是一个元组。**

➋ **kwargs 是一个字典。**

在函数定义签名中，*和** *打包*参数，而在函数调用中，它们*解包*参数。在函数调用中解包参数等同于在 R 中使用 do.call()：

def my_func(a, b, c):

print(a, b, c)

args = (1, 2, 3)

my_func(*args)

1 2 3

kwargs = {"a": 1, "b": 2, "c": 3}

my_func(**kwargs)

1 2 3

## A.5 使用 class 定义类

有人可能会争论，在 R 中，代码的主要组成单位是函数，在 Python 中，它是类。你可以成为一个非常高效的 R 用户，而从不使用 R6、引用类或类似的 R 等价物来实现 Python 类的面向对象风格。

然而，在 Python 中，理解类对象如何工作的基础知识是必不可少的，因为类是你如何组织和查找 Python 方法的方式（与 R 的方法相比，在 R 中，方法是通过从通用方法分派来找到的）。幸运的是，类的基础知识是可以理解的。

如果这是您第一次接触面向对象编程，不要感到 intimidated。我们将从构建一个简单的 Python 类开始作为演示：

class MyClass:

pass➊

MyClass

<class '__main__.MyClass'>

type(MyClass)

<class 'type'>

instance = MyClass()

instance

<__main__.MyClass object at 0x7f5e30fc7790>

type(instance)

<class '__main__.MyClass'>

➊ **pass 意味着什么都不做。**

类似于 def 语句，class 语句绑定了一个新的可调用符号，MyClass。首先注意到强命名约定：类通常是 CamelCase，函数通常是 snake_case。在定义 MyClass 之后，你可以与之交互，并且看到它的类型为‘type’。调用 MyClass()创建了一个类的新对象*instance*，它的类型是‘MyClass’（现在忽略 __main__.前缀）。实例打印出其内存地址，这是一个强烈的暗示，表明通常会管理许多类的实例，并且该实例是可变的（默认情况下是就地修改的）。

在第一个例子中，我们定义了一个空类，但当我们检查它时，我们会发现它已经带有一堆属性（在 Python 中，dir()等同于 R 中的 names()）：

dir(MyClass)

['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__',

'__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__',

'__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__',

'__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__',

'__str__', '__subclasshook__', '__weakref__']

### A.5.1 所有下划线都是什么？

Python 通常通过双下划线包裹名称来表示某些特殊性，而常见的双下划线包裹的标记通常被称为*dunder*。“特殊”不是一个技术术语；它只是表示该标记调用了 Python 语言的一个特性。一些 dunder 标记仅仅是代码作者可以插入特定语法糖的方式；其他的是解释器提供的值，否则可能很难获得；还有一些是用于扩展语言接口（例如，迭代协议）；最后，少数一小部分 dunder 真的很难理解。幸运的是，作为一个希望通过 reticulate 使用一些 Python 特性的 R 用户，你只需要了解一些易于理解的 dunder。

阅读 Python 代码时最常见的特殊方法是 __init__()。这是一个在调用类构造函数时调用的函数，也就是在类被实例化时。它用于初始化新的类实例。（在非常复杂的代码库中，您可能还会遇到定义了 __new__() 的类；这是在调用 __init__() 之前调用的。）

class MyClass：

print("MyClass 的定义主体正在被评估")➊

def __init__(self)：

print(self, "正在初始化")

MyClass 的定义主体正在被评估

instance = MyClass()

<__main__.MyClass object at 0x7f5e30fcafd0> 正在初始化➋

print(instance)

<__main__.MyClass object at 0x7f5e30fcafd0>➋

instance2 = MyClass()

<__main__.MyClass object at 0x7f5e30fc7790> 正在初始化➌

print(instance2)

<__main__.MyClass object at 0x7f5e30fc7790>➌

➊ **注意这是在类第一次被定义时评估的。**

➋ **注意`instance`和`self`在 __init__() 方法中的相同内存地址。**

➌ **新实例，新内存地址**

请注意以下几点：

+   class 语句采用由共同缩进级别定义的代码块。代码块与任何其他接受代码块的表达式具有完全相同的语义，如 if 和 def。类的主体仅在第一次创建类构造函数时被评估一次。请注意，此处定义的任何对象都将由类的所有实例共享！

+   __init__() 只是一个普通的函数，使用 def 定义，与任何其他函数一样，只是在类主体内部定义。

+   __init__() 接受一个参数：self。self 是被初始化的类实例（注意 self 和实例之间的内存地址相同）。还要注意，当调用 MyClass() 创建类实例时，我们没有提供 self；语言会将 self 插入到函数调用中。

+   每次创建新实例时都会调用 __init__()。

在类代码块中定义的函数称为*方法*，方法的重要之处在于每次从类实例中调用它们时，实例都会作为第一个参数插入到函数调用中。这适用于类中定义的所有函数，包括特殊方法。唯一的例外是，如果函数被装饰为 @classmethod 或 @staticmethod：

class MyClass：

def a_method(self)：

print("MyClass.a_method() 被调用时使用了", self)

instance = MyClass()

instance.a_method()

MyClass.a_method() 被调用时使用了<__main__.MyClass object at 0x7f5e30fcadf0>：

MyClass.a_method()➊

在 py_call_impl(callable, dots$args, dots$keywords) 中出现错误：

![Image](img/common01.jpg) TypeError: a_method() 缺少 1 个必需的位置参数：'self'

MyClass.a_method(instance)➋

MyClass.a_method() 被调用时使用了<__main__.MyClass object at 0x7f5e30fcadf0>

➊ **错误：缺少必需的参数 self**

➋ **与 instance.a_method() 相同**

其他值得了解的特殊方法有：

+   __getitem__—提取切片时调用的函数（相当于在 R 中定义 S3 方法）。

+   __getattr__—使用.访问属性时调用的函数（相当于在 R 中定义$ S3 方法）。

+   __iter__ 和 __next__—由 for 循环调用的函数。

+   __call__—当类实例被像函数一样调用时调用（例如，instance()）。

+   __bool__—由 if 和 while 调用（相当于 as.logical()在 R 中，但只返回标量，而不是向量）。

+   __repr__ 和 __str__—用于格式化和漂亮打印的函数（类似于 R 中的 format()、dput()和 print()方法）。

+   __enter__ 和 __exit__—由 with 语句调用的函数。

+   许多内置的 Python 函数只是调用 dunder 的语法糖。例如，调用 repr(x)与 x.__repr__()是相同的（参见[`docs.python.org/3/library/functions.html`](https://docs.python.org/3/library/functions.html)）。其他的内置函数，比如 next()、iter()、str()、list()、dict()、bool()、dir()、hash()等等，都是调用 dunder 的语法糖！

### A.5.2 迭代器，重新审视

现在我们已经掌握了类的基础知识，是时候重新审视迭代器了。首先，一些术语：

+   *可迭代对象*—可以被迭代的东西。具体来说，是定义了一个 __iter__ 方法的类，其作用是返回一个*迭代器*。

+   *迭代器*—一种进行迭代的东西。具体来说，是定义了一个 __next__ 方法的类，其作用是每次调用时返回下一个元素，然后在耗尽时引发 StopIteration 异常。常见的情况是看到既是可迭代对象又是迭代器的类，其中 __iter__ 方法只是一个返回 self 的存根。这里是 Python 中 range()的自定义可迭代/迭代器实现（类似于 R 中的 seq()）：

class MyRange:

def __init__(self, start, end):

self.start = start

self.end = end

def __iter__(self):

self._index = self.start - 1➊

return self

def __next__(self):

if self._index < self.end:

self._index += 1➋

return self._index

else:

raise StopIteration

for x in MyRange(1, 3):

print(x)

1

2

3

➊ **重置我们的计数器。**

➋递增 1。

手动执行 for 循环的操作：

r = MyRange(1, 3)

it = iter(r)

next(it)

1

next(it)

2

next(it)

3

next(it)

在 py_call_impl(callable, dots$args, dots$keywords)中的错误：StopIteration

## A.6 使用 yield 定义生成器

生成器是特殊的 Python 函数，其中包含一个或多个 yield 语句。只要在传递给 def 的代码块中包含 yield，语义就会发生根本变化。您不再只是定义一个普通的函数，而是一个生成器构造函数！反过来，调用生成器构造函数会创建一个生成器对象，它只是另一种类型的迭代器。这里有一个例子：

def my_generator_constructor():

yield 1

yield 2

yield 3

乍一看，它看起来像一个普通的函数：

my_generator_constructor

<function my_generator_constructor at 0x7f5e30fab670>

type(my_generator_constructor)

<class 'function'>

但是调用它会返回一些特殊的东西，一个*生成器对象*：

my_generator = my_generator_constructor()

my_generator

<generator object my_generator_constructor at 0x7f5e3ca52820>

type(my_generator)

<class 'generator'>

生成器对象既是可迭代的，也是迭代器。它的 __iter__ 方法只是返回 self 的一个存根：

iter(my_generator) == my_generator == my_generator.__iter__()

True

像任何其他迭代器一样逐步执行它：

next(my_generator)

1

my_generator.__next__()➊

2

next(my_generator)

3

next(my_generator)

py_call_impl(callable, dots$args, dots$keywords)中的错误：StopIteration

➊ **next(x) is just sugar for calling the dunder x.__next__().**

遇到 yield 就像按下函数执行的暂停按钮：它保留函数体中的所有状态，并将控制返回给迭代生成器对象的任何东西。对生成器对象调用 next()会恢复函数体的执行，直到下一个 yield 被遇到或函数完成。您可以使用 coro::generator()在 R 中创建生成器。

## A.7 迭代结束语

迭代在 Python 语言中已经深深融入，R 用户可能会对 Python 中的事物是可迭代的、迭代器或者在幕后由迭代器协议支持的方式感到惊讶。例如，内置的 map()（相当于 R 的 lapply()）产生一个迭代器，而不是一个列表。类似地，像(elem for elem in x)的元组推导式会产生一个迭代器。大多数涉及文件的功能都是迭代器。

每当您发现一个迭代器不方便时，您可以使用 Python 内置的 list()或 R 中的 reticulate::iterate()将所有元素材化为列表。此外，如果您喜欢 for 的可读性，您可以使用类似 Python 的 for 的语义来利用 coro::loop()。

## A.8 导入和模块

在 R 中，作者可以将其代码捆绑成可共享的扩展，称为 R 包，而 R 用户可以通过 library()或::访问 R 包中的对象。在 Python 中，作者将代码捆绑成*模块*，用户使用 import 来访问模块。考虑以下行：

import numpy

这个语句让 Python 去文件系统找到一个名为 numpy 的已安装 Python 模块，加载它（通常意味着：评估其 __init__.py 文件并构造一个模块类型对象），并将其绑定到符号 numpy。在 R 中，这个最接近的相当于可能是：

dplyr <- loadNamespace("dplyr")

### A.8.1 模块存储在哪里？

在 Python 中，模块搜索的文件系统位置可以从 sys.path 找到并（修改）。这相当于 R 的.lib-Paths()。sys.path 通常包含当前工作目录的路径，包含内置标准库的 Python 安装路径，管理员安装的模块，用户安装的模块，像 PYTHONPATH 这样的环境变量的值，以及当前 Python 会话中其他代码直接对 sys.path 进行的任何修改（尽管在实践中这相对较少见）：

import sys

sys.path

['',➊

'/home/tomasz/.pyenv/versions/3.9.6/bin',

'/home/tomasz/.pyenv/versions/3.9.6/lib/python39.zip',

'/home/tomasz/.pyenv/versions/3.9.6/lib/python3.9',

'/home/tomasz/.pyenv/versions/3.9.6/lib/python3.9/lib-dynload',➋

'/home/tomasz/.virtualenvs/r-reticulate/lib/python3.9/site-packages',➌

'/home/tomasz/opt/R-4.1.2/lib/R/site-library/reticulate/python',➍

'/home/tomasz/.virtualenvs/r-reticulate/lib/python39.zip',

'/home/tomasz/.virtualenvs/r-reticulate/lib/python3.9',

'/home/tomasz/.virtualenvs/r-reticulate/lib/python3.9/lib-dynload']➎

➊ **当前目录通常位于模块搜索路径上。**

➋ **Python 标准库和内建函数**

➌ **reticulate 代理**

➍ **其他安装的 Python 包（例如，通过 pip 安装）**

➎ **更多的标准库和内建函数，这次来自虚拟环境**

你可以通过访问 dunder __path__ 或 __file__（在排除安装问题时特别有用）来查看模块是从哪里加载的：

import os

os.__file__

'/home/tomasz/.pyenv/versions/3.9.6/lib/python3.9/os.py'➊

numpy.__path__

['/home/tomasz/.virtualenvs/r-reticulate/lib/python3.9/site-packages/numpy']➋

➊ **os 模块在这里定义。它只是一个普通的文本文件；看一眼吧！**

➋ **我们导入的 numpy 模块在这里定义。它是一个有很多内容的目录；浏览一下！**

一旦加载了模块，就可以使用 .（相当于::，或者可能是 $.environment，在 R 中）访问模块中的符号：

numpy.abs(-1)

1

还有一种特殊的语法来指定模块在导入时绑定到的符号，以及仅导入一些特定的符号：

import numpy➊

import numpy as np➋

np 是 numpy➌

从 numpy 导入 abs➍

abs 是 numpy.abs➎

从 numpy 导入 abs 作为 abs2➏

abs2 是 numpy.abs➐

➊ **导入并绑定到符号 'numpy'。**

➋ **导入并绑定到自定义符号 'np'。**

➌ **测试是否相同，类似于 R 的 identical(np, numpy)。返回 True。**

➍ **仅导入 numpy.abs，并绑定到 abs。**

➎ **True**

➏ **仅导入 numpy.abs，并绑定到 abs2。**

➐ **True**

如果你正在寻找 R 的 library() 的 Python 等效，它使包的所有导出符号都可用，可能会使用 import 与 * 通配符，尽管这样做相对较少见。* 通配符将扩展为包含模块中的所有符号，或者如果定义了 __all__，则包含在其中列出的所有符号：

from numpy import *

Python 不像 R 那样区分包导出和内部符号。在 Python 中，所有模块符号都是平等的，尽管有一种命名约定，即打算为内部符号的符号以单个下划线作为前缀。（两个前导下划线会调用一个称为“名称修饰”的高级语言特性，这超出了本介绍的范围。）

如果你正在寻找 Python 中与 import 语法等效的 R 语法，你可以像这样使用 envir::import_from()：

library(envir)

import_from(keras::keras$applications$efficientnet,

decode_predictions, preprocess_input,

new_model = EfficientNetB4)

model <- new_model(include_top = TRUE, weights='imagenet')

predictions <- input_data %>%

preprocess_input()

%>% predict(model, .) %>%

decode_predictions()

## A.9 整数和浮点数

R 用户通常不需要了解整数和浮点数之间的区别，但在 Python 中情况并非如此。如果这是你第一次接触数字数据类型，以下是必备知识：

+   整数类型只能表示像 2 或 3 这样的整数，不能表示像 2.3 这样的浮点数。

+   浮点类型可以表示任何数字，但存在一定程度的不精确性

在 R 中，像 3 这样写一个裸的文字数会产生一个浮点类型，而在 Python 中，它会产生一个整数。你可以通过在 R 中附加一个 L 来产生一个整数文字，例如 3L。许多 Python 函数期望整数，并在提供浮点数时发出错误。例如，假设我们有一个期望整数的 Python 函数：

def a_strict_Python_function(x):

assert isinstance(x, int), "x 不是整数"

print("耶！x 是一个整数")

当从 R 中调用时，必须确保以整数形式调用：

library(reticulate)

py$a_strict_Python_function(3)➊

py$a_strict_Python_function(3L)

py$a_strict_Python_function(as.integer(3))➋

➊ **错误："AssertionError: x 不是整数"**

➋ **成功**

## A.10 R 向量怎么办？

R 是一个以数值计算为首要目标的语言。数值向量数据类型已经深深地融入到 R 语言中，以至于语言甚至不区分标量和向量。相比之下，Python 中的数值计算能力通常由第三方包（在 Python 术语中称为*模块*）提供。

在 Python 中，numpy 模块通常用于处理数据的连续数组。与 R 数值向量最接近的等价物是 1D NumPy 数组，有时是标量数值的列表（一些 Python 爱好者可能会认为这里应该使用 array.array()，但实际 Python 代码中很少遇到，因此我们不再进一步讨论）。

NumPy 数组与 TensorFlow 张量非常相似。例如，它们共享相同的广播语义和非常相似的索引行为。NumPy API 非常广泛，教授完整的 NumPy 接口超出了本入门教程的范围。然而，值得指出一些对习惯于 R 数组的用户可能构成潜在绊脚石的地方：

+   当对多维 NumPy 数组进行索引时，可以省略尾部维度，并且会被隐式地视为缺失。其结果是对数组进行迭代意味着对第一维进行迭代。例如，这会对矩阵的行进行迭代

import numpy as np

m = np.arange(12).reshape((3,4))

m

array([[ 0, 1, 2, 3],

[ 4, 5, 6, 7],

[ 8, 9, 10, 11]])

m[0, :]➊

array([0, 1, 2, 3])

m[0]➋

array([0, 1, 2, 3])

for row in m:

print(row)

[0 1 2 3]

[4 5 6 7]

[ 8 9 10 11]

➊ **第一行**

➋ **也是第一行**

+   许多 NumPy 操作会直接修改数组！这让 R 用户（和 TensorFlow 用户）感到惊讶，他们习惯于 R（和 TensorFlow）的按需复制语义的便利性和安全性。不幸的是，没有简单的方案或命名约定可以依赖于快速确定特定方法是直接修改还是创建新数组副本。唯一可靠的方法是查阅文档（请参阅 [`mng.bz/mORP`](http://mng.bz/mORP)），并在 reticulate::repl_python() 中进行小实验。

## A.11 装饰器

装饰器只是接受一个函数作为参数并通常返回另一个函数的函数。任何函数都可以使用 @ 语法调用装饰器，这只是这个简单动作的语法糖：

def my_decorator(func):

func.x = "一个装饰器通过添加属性 `x` 修改了这个函数"

return func

@my_decorator

def my_function(): pass

def my_function(): pass

my_function = my_decorator(my_function)➊

➊ **@decorator 只是这一行的花哨语法。**

你可能经常遇到的一个装饰器是 @property，当访问属性时自动调用类方法（类似于 R 中的 makeActiveBinding()）：

from datetime import datetime

class MyClass:

@property

def a_property(self):

return f"`a_property` was accessed at {datetime.now().strftime('%X')}"

instance = MyClass()

instance.a_property

'`a_property` was accessed at 10:01:53 AM'

你可以使用 %<-active%（或 mark_active()）将 Python 的 @property 翻译为 R，就像这样：

import_from(glue, glue)

MyClass %py_class% {

a_property %<-active% function()

glue("`a_property` was accessed at {format(Sys.time(), '%X')}")

}

instance <- MyClass()

instance$a_property

[1] "`a_property` was accessed at 10:01:53 AM"

Sys.sleep(1)

instance$a_property

[1] "`a_property` was accessed at 10:01:54 AM"

## A.12 with 和上下文管理

任何定义了 __enter__ 和 __exit__ 方法的对象都实现了“上下文”协议，并且可以传递给 with。例如，这里是一个自定义的上下文管理器的实现，它临时更改了当前工作目录（相当于 R 的 withr::with_dir()）：

from os import getcwd, chdir

class wd_context:

def __init__(self, wd):

self.new_wd = wd

def __enter__(self):

self.original_wd = getcwd()

chdir(self.new_wd)

def __exit__(self, *args):➊

chdir(self.original_wd)

getcwd()

'/home/tomasz/deep-learning-w-R-v2/manuscript'

with wd_context("/tmp"):

print("在上下文中，wd 是：", getcwd())

在上下文中，wd 是： /tmp

getcwd()

'/home/tomasz/deep-learning-w-R-v2/manuscript'

➊ **__exit__ 接受一些通常被忽略的附加参数。**

## A.13 进一步学习

希望这篇关于 Python 的简短入门对于自信地阅读 Python 文档和代码，以及通过 reticulate 从 R 使用 Python 模块提供了良好的基础。当然，关于 Python 还有很多，很多需要学习的地方。在谷歌上搜索有关 Python 的问题可靠地会出现大量结果页面，但并不总是按照最有用的顺序排序。针对初学者的博客文章和教程可能很有价值，但请记住 Python 的官方文档通常是非常出色的，当您有问题时它应该是您的首选目的地：

+   [`docs.Python.org/3/`](https://www.docs.Python.org/3/)

+   [`docs.Python.org/3/library/index.htm`](https://www.docs.Python.org/3/library/index.htm)

要更全面地学习 Python，内置的官方教程也是非常出色和全面的（但需要投入一定的时间来获得价值）：[`docs.Python.org/3/tutorial/index.html`](https://www.docs.Python.org/3/tutorial/index.html)。

最后，请不要忘记通过在 reticulate::repl_python()中进行小型实验来巩固您的理解。

谢谢您的阅读！
