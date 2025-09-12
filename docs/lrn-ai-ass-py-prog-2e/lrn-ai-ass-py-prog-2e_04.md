# 第五章：阅读 Python 代码：第二部分

### 本章涵盖

+   使用循环重复所需的代码次数

+   使用缩进来告诉 Python 哪些代码属于一组

+   构建字典来存储相关值的对

+   设置文件以读取和处理数据

+   使用模块在新的领域工作

在第四章中，我们探讨了你在继续你的编程之旅时将经常看到的五个 Python 特性：函数、变量、条件语句（`if` 语句）、字符串和列表。你需要了解这些特性来阅读代码，我们也解释了为什么无论是否使用 Copilot，能够阅读代码都很重要。

在本章中，我们将继续介绍五个更多的 Python 特性，这将使我们的前 10 个特性更加完整。与第四章一样，我们将通过我们自己的解释、Copilot 的解释以及在 Python 提示符下的实验来做到这一点。

## 5.1 你需要知道的 10 个编程特性：第二部分

本节详细介绍了你需要知道的下一个五个顶级编程特性。让我们从上一章留下的地方继续，即第 6 个特性：循环。

### 5.1.1 #6\. 循环

循环允许计算机根据需要重复执行相同的代码块。如果我们前 10 个编程特性中的任何一个能体现为什么计算机对我们完成工作如此有用，那就是这个特性。如果没有循环的能力，我们的程序通常会按顺序逐行执行。当然，它们仍然可以调用函数并使用 `if` 语句来做出决定，但程序完成的工作量将与我们编写的代码量成比例。但循环不是这样：一个循环可以轻松处理成千上万的值。

有两种类型的循环：`for` 循环和 `while` 循环。一般来说，当我们知道循环需要运行多少次时，我们使用 `for` 循环；当我们不知道时，我们使用 `while` 循环。例如，在第三章中，我们的 `best_word` 函数（如列表 5.1 所示）使用了一个 `for` 循环，因为我们知道循环需要运行多少次：对 `word_list` 中的每个单词运行一次！但在 `get_strong_password` 中，我们将在列表 5.4 中再次看到它，我们使用了一个 `while` 循环，因为我们不知道用户在输入一个强密码之前会输入多少个坏密码。我们将从 `for` 循环开始，然后转向 `while` 循环。

##### 列表 5.1 来自第三章的 `best_word` 函数

```py
def best_word(word_list):
 """
 word_list is a list of words.

 Return the word worth the most points.
 """
    best_word = ""
    best_points = 0
    for word in word_list:       **#1
        points = num_points(word)
        if points > best_points:
            best_word = word
            best_points = points
    return best_word**
```

**#1 这是一个 for 循环的例子。** **`for` 循环允许我们访问字符串或列表中的每个值。让我们先从一个字符串开始尝试：

```py
>>> s = 'vacation'
>>> for char in s:       #1
...     print('Next letter is', char)    #2
...
Next letter is v
Next letter is a
Next letter is c
Next letter is a
Next letter is t
Next letter is i
Next letter is o
Next letter is n
```

#1 这将重复缩进的代码，每次对应字符串 s 中的一个字符。

#2 因为“vacation”有八个字母，所以这段代码将运行八次。

注意，我们不需要为 `char` 赋值语句。这是因为它是一个特殊的变量，称为循环变量，它由 `for` 循环自动管理。`char` 代表字符，这是人们用来命名循环变量的一个非常流行的名字。`char` 变量会自动分配字符串中的每个字符。在谈论循环时，我们经常使用单词 *迭代* 来指代每次通过循环执行的代码。例如，我们可以说在第一次迭代中，`char` 指的是 `v`；在第二次迭代中，它指的是 `a`；依此类推。注意，就像函数和 `if` 语句一样，我们为构成循环的代码有缩进。在这个循环的主体中，我们只有一行代码，但就像函数和 `if` 语句一样，我们也可以有更多。

让我们看看一个 `for` 循环在列表上的示例（列表 5.2），演示我们可以像处理字符串的每个值一样处理列表的每个值。我们还会在循环中放入两行代码，而不是一行，以演示这是如何工作的。

##### 列表 5.2 使用 `for` 循环的示例

```py
>>> lst = ['cat', 'dog', 'bird', 'fish']
>>> for animal in lst:               #1
...     print('Got', animal)       #2
...     print('Hello,', animal)   ** #2
...
Got cat
Hello, cat
Got dog
Hello, dog
Got bird
Hello, bird
Got fish
Hello, fish**
```

**#1 第一个是一个列表，所以这是一个列表上的 for 循环。

#2 这段代码在每次迭代时运行。**  **列表 5.2 中的代码只是循环遍历列表的一种方式。`for` `animal` `in` `lst` 的方法在每次通过循环时将变量 `animal` 赋值为列表中的下一个值。作为替代，您可以使用索引来访问列表中的每个元素。为此，我们需要了解内置的 `range` 函数。

`range` 函数可以给你一个范围内的数字。我们可以提供一个起始数字和一个结束数字，它将生成从起始数字开始，但不包括结束数字的范围。要查看 `range` 生成的数字，我们需要在它周围放置 `list` 函数。以下是一个使用 `range` 的示例：

```py
>>> list(range(3, 9))     #1
[3, 4, 5, 6, 7, 8]
```

#1 生成从 3 到 8 的范围（不是 3 到 9！）

注意，它从值 `3` 开始，包括 `3` 和 `8` 之间的所有值。也就是说，它包括从起始值 `3` 到，但不包括，结束值 `9` 的所有数字。

那么，`range` 如何帮助我们编写循环呢？嗯，而不是在范围中硬编码像 3 和 9 这样的数字，我们可以包括字符串或列表的长度，如下所示：

```py
>>> lst
['cat', 'dog', 'bird', 'fish'] 
>>> list(range(0, len(lst)))      #1
[0, 1, 2, 3]
```

#1 从 0 开始，直到但不包括 1st 的长度。

注意，这里的范围值是 0、1、2、3，这是我们的 `lst` 列表的有效索引！因此，我们可以使用 `range` 来控制 `for` 循环，这将使我们能够访问字符串或列表中的每个有效索引。

我们可以使用 `range` 在列表 5.2 中执行相同的任务。查看列表 5.3 以获取新代码。

##### 列表 5.3 使用 `for` 循环和 `range` 的循环示例

```py
>>> for index in range(0, len(lst)):        #1
...     print('Got', lst[index])        #2
...     print('Hello,', lst[index])    ** #2
...
Got cat
Hello, cat
Got dog
Hello, dog
Got bird
Hello, bird
Got fish
Hello, fish**
```

**#1 使用 `range` 函数的 for 循环

#2 使用索引变量对列表进行索引**  **在这里，我们使用了一个名为`index`的变量，但你也经常会看到人们为了简单起见只使用`i`。该变量将在循环的第一次迭代时被赋予`0`的值，第二次迭代时为`1`，第三次迭代时为`2`，最后一次迭代时为`3`。它停止在`3`，因为列表的长度是 4，而`range`在它之前停止。使用列表的索引，代码获取第一个元素，然后是第二个，然后是第三个，最后是第四个，使用递增的索引。我们也可以不写`0`来编写`for`循环；`range`将假设我们想要从`0`到提供的值的值，如下所示：

```py
for index in range(len(lst)):    #1
    print('Got', lst[index])
    print('Hello,', lst[index])
```

#1 使用一个参数时，`range`假设我们要从 0 开始。

我们在这里就停止`for`循环的讨论。但我们还没有结束循环的讨论，因为还有另一种类型的循环需要我们讨论：`while`循环。

当我们不知道要循环多少次时，我们会使用`while`循环。这种情况的一个很好的例子是在第三章的`get_strong_password`函数中。我们在这里将其代码作为列表 5.4 重现。 

##### 列表 5.4 第三章的`get_strong_password`函数

```py
def get_strong_password():
 """
 Keep asking the user for a password until it’s a strong password,
 and return that strong password.
 """
    password = input("Enter a strong password: ")
    while not is_strong_password(password):           #1
        password = input("Enter a strong password: ")
    return password
```

#1 当密码不够强大时持续循环

我们不知道用户会输入多少个密码，直到他们输入一个强大的密码。是第一次尝试、第二次尝试，还是第 50 次尝试？谁知道。这就是为什么我们需要一个`while`循环：它确保用户会不断被要求输入，直到他们提供一个强大的密码。

要编写一个`while`循环，我们需要在`while`后面跟一个布尔条件。如果它是`True`，Python 将运行循环的一次迭代。如果在那次迭代之后，那个布尔条件仍然是`True`，那么它将运行另一次迭代。它将一直这样做，直到条件变为`False`，此时循环终止。

现在，我们有了足够的信息来理解我们的`get_strong_password`函数的代码。通过查看代码，我们可以看到它首先使用`input`函数来要求用户输入密码。然后，有一个`while`循环。这个循环会一直持续，直到密码不是一个强大的密码。布尔条件`not is_strong_password(password)`等同于写作`is_strong_password(password) == False`。为了避免无限循环并要求用户再次尝试，我们在循环内部又使用了另一个`input`。我们需要在循环上面的`input`调用，以从用户那里获取第一次尝试。如果没有这个，`password`变量就不会存在，而`while`循环行会出错，因为它没有`password`值传递给`is_strong_password`函数。

不要忘记，我们可以向 Copilot 请求对这个代码的解释，使用我们在第四章第 4.2 节学到的知识。当你觉得你并不完全理解代码正在做什么或它是如何工作时，这很有用。这是我们从 Copilot 那里收到的解释：

这段 Python 代码定义了一个名为`get_strong_password`的函数。这个函数的目的是交互式地提示用户输入密码，并持续询问，直到用户提供一个被认为是强密码的密码。构成强密码的标准在此函数中未定义，而是由另一个名为`is_strong_password`的函数确定。

函数开始时使用 Python 的内置`input`函数提示用户输入密码。然后，输入的密码存储在变量`password`中。

接下来，函数进入一个`while`循环。这个循环的条件是`not` `is_strong_` `password(password)`。这意味着只要`is_strong_password`函数在用当前密码调用时返回`False`，循环就会继续执行。换句话说，循环会一直运行，直到用户输入一个被认为是强密码的密码。

在循环内部，函数再次提示用户输入密码。新密码替换了之前存储在`password`变量中的旧密码。如果这个新密码仍然不够强，循环会继续提示用户。

一旦用户输入了一个被认为是强密码（即`is_strong_password`(password)返回`True`），`while`循环的条件变为`False`，循环结束。然后函数返回用户输入的强密码。

这个函数是使用`while`循环反复提示用户输入直到输入满足某些条件的一个很好的例子。然而，强密码的实际标准并未在此函数中定义，需要在`is_strong_password`函数中实现。

这个解释非常详细地描述了函数，与我们刚刚给出的描述相匹配！它还包含了一些有用的 Python 通用信息，包括在最后一段中提醒我们为什么我们使用`while`循环。解释还提到我们需要`is_strong_password`函数来告诉我们密码是否强大；我们没有强调这个函数，这就是为什么 Copilot 告诉我们需要实现它。

##### Copilot 解释可能会出错

我们选择之前的 Copilot 解释，因为它是在我们要求 Copilot 解释代码三次后给出的最佳答案。它给出的其中一个答案听起来相当可信，直到它开始谈论不存在的函数。我们相信，如果您多次运行并寻找共同的想法，这些解释可以作为学习辅助工具很有帮助，但本章的主要目标是为您提供理解何时出错所需的工具。

我们鼓励您继续使用 Copilot 解释，如果您感兴趣，可以要求 Copilot 解释您仍然好奇的任何前几章的代码。再次提醒，这些解释可能会出错，因此您应该要求 Copilot 提供多个解释以减少对单个错误解释的依赖。

就像现在与任何与 AI 编码助手相关的事情一样，它们可能会出错。但我们在这里给出解释，因为我们认为 Copilot 的这个功能现在是一个潜在的有力教学资源，而且随着 Copilot 的改进，这一点将变得更加真实。

在我们不知道迭代次数的情况下，我们应该使用 `while` 循环。但即使我们知道迭代次数，我们也可以使用 `while` 循环。例如，我们可以使用 `while` 循环来处理字符串中的字符或列表中的值。我们有时会在 Copilot 生成的代码中看到它这样做，尽管使用 `for` 循环可能更好。例如，我们可以使用 `while` 循环来处理我们之前提到的 `animals` 列表中的动物，如下所示。但这会多做一些工作！

##### 列表 5.5 使用 `while` 循环的循环示例

```py
>>> lst
['cat', 'dog', 'bird', 'fish'] 
>>> index = 0
>>> while index < len(lst):        #1
...     print('Got', lst[index])
...     print('Hello,', lst[index])
...     index += 1          #2
...
Got cat
Hello, cat
Got dog
Hello, dog
Got bird
Hello, bird
Got fish
Hello, fish
```

#1 `len` 函数告诉我们字符串的长度，这也是我们想要的迭代次数。

#2 这是常见的错误，很多人都会犯这个错误！

如果没有 `index` `+=` `1`，我们就永远不会增加字符串中的索引，我们会不断地输出第一个值的详细信息。这被称为*无限循环*。如果你回想一下我们是如何编写 `for` 循环的，你会发现我们不必手动增加任何索引变量。出于这些原因，许多程序员在可能的情况下更喜欢使用 `for` 循环。我们不必在 `for` 循环中手动跟踪任何索引，因此我们自动避免了某些类型的索引问题和无限循环。

### 5.1.2 #7\. 缩进

在 Python 代码中，缩进至关重要，因为 Python 使用它来确定哪些代码行属于一起。这就是为什么，例如，我们总是在函数内部的代码行、`if` 语句的各个部分以及 `for` 或 `while` 循环的代码中进行缩进。这不仅仅是格式化得更好：如果我们缩进错误，那么代码也会出错。例如，假设我们想要询问用户当前的小时，然后根据是早上、下午还是晚上输出一些文本：

+   如果是早上，我们想要输出“早上好！”和“祝您有个愉快的一天。”

+   如果是下午，我们想要输出“下午好！”

+   如果是晚上，我们想要输出“晚上好！”和“祝您有个美好的夜晚。”

看看我们编写的以下代码，并尝试找出缩进的问题：

```py
hour = int(input('Please enter the current hour from 0 to 23: '))

if hour < 12:
    print('Good morning!')
    print('Have a nice day.')
elif hour < 18:
    print('Good afternoon!')
else:
    print('Good evening!')
print('Have a good night.')     #1
```

#1 这一行没有缩进。

问题在于最后一行：它没有缩进，但它应该缩进！因为它没有缩进，所以无论用户输入哪个小时，我们都会输出 `Have` `a` `good` `night.`。我们需要缩进它，使其成为 `if` 语句的 `else` 部分的一部分，确保它只在晚上执行。

无论何时编写代码，我们都需要使用多级缩进来表达哪些代码片段与函数、`if`语句、循环等相关联。例如，当我们编写函数头时，我们需要将函数头下面的所有相关代码缩进。一些语言使用括号（例如{}）来显示这一点，但 Python 只是缩进。如果你已经在函数体（一个缩进）中编写了一个循环，那么你将需要再次缩进（两个缩进）以缩进循环体，依此类推。

回顾第三章中的函数，我们可以看到这一点。例如，在我们的`larger`函数（重新打印为列表 5.6）中，整个函数体都是缩进的，但在`if`语句的`if`部分和`else`部分有进一步的缩进。

##### 列表 5.6 比较两个值大小的函数

```py
def larger(num1, num2):
    if num1 > num2:       #1
        return num1    #2
    else:                 #3
        return num2       #4
```

#1 这显示了函数体的单级缩进。

#2 这显示了函数体和 if 语句体的双重缩进。

#3 这显示了函数体的单级缩进。

#4 这显示了函数体和 else 语句体的双重缩进。

接下来，考虑我们之前在列表 5.4 中查看的`get_strong_password`函数：通常，函数中的所有内容都是缩进的，但`while`循环体的缩进更深。

在我们`num_points`函数的第一版中（此处从第三章的列表 5.7 中复制），甚至还有更多级别的缩进。这是因为，在遍历单词每个字符的`for`循环内部，我们有一个`if`语句。正如我们所学的，`if`语句的每一部分都需要缩进，从而导致额外的缩进级别。

##### 列表 5.7 `num_points`函数

```py
def num_points(word): 
 """ 
 Each letter is worth the following points: 
 a, e, i, o, u, l, n, s, t, r: 1 point 
 d, g: 2 points 
 b, c, m, p: 3 points 
 f, h, v, w, y: 4 points 
 k: 5 points 
 j, x: 8 points 
 q, z: 10 points 

 word is a word consisting of lowercase characters. 
 Return the sum of points for each letter in word. 
 """
    points = 0
    for char in word:            #1
        if char in "aeioulnstr":     #2
            points += 1          #3
        elif char in "dg":
            points += 2
        elif char in "bcmp":
            points += 3
        elif char in "fhvwy":
            points += 4
        elif char == "k":
            points += 5
        elif char in "jx":
            points += 8
        elif char in "qz":
            points += 10
    return points
```

#1 这是为了位于函数内部而缩进的。

#2 这再次缩进，以便位于 for 循环内部。

#3 这再次缩进，以便位于 if 语句内部。

在`is_strong_password`函数中也有额外的缩进（此处从第三章的列表 5.8 中复制），但这只是为了将一条超长的代码行扩展到多行。注意，这些行以`\`结尾，这是我们可以在下一行继续代码行的字符。

##### 列表 5.8 `is_strong_password`函数

```py
def is_strong_password(password):
 """
 A strong password has at least one uppercase character,
 at least one number, and at least one punctuation.

 Return True if the password is a strong password, 
 False if not.
 """
    return any(char.isupper() for char in password) and \     #1
           any(char.isdigit() for char in password) and \     #2
           any(char in string.punctuation for char in password)
```

#1 这行以反斜杠结尾，以继续语句。

#2 缩进不是必需的，但有助于在视觉上布局单行返回语句。

类似地，在我们的`num_points`函数的第二版中（此处从第三章的列表 5.9 中复制）也有一些进一步的缩进，但这只是为了将字典扩展到多行，使其更易于阅读。

##### 列表 5.9 `num_points`的替代解决方案

```py
 def num_points(word): 
 """ 
 Each letter is worth the following points: 
 a, e, i, o, u, l, n, s, t, r: 1 point 
 d, g: 2 points 
 b, c, m, p: 3 points 
 f, h, v, w, y: 4 points 
 k: 5 points 
 j, x: 8 points 
 q, z: 10 points 

 word is a word consisting of lowercase characters. 
 Return the sum of points for each letter in word. 
 """ 
    points = {'a': 1, 'e': 1, 'i': 1, 'o': 1, 'u': 1, 'l': 1,     #1
              'n': 1, 's': 1, 't': 1, 'r': 1,        #2
              'd': 2, 'g': 2,
              'b': 3, 'c': 3, 'm': 3, 'p': 3,
              'f': 4, 'h': 4, 'v': 4, 'w': 4, 'y': 4,
              'k': 5,
              'j': 8, 'x': 8,
              'q': 10, 'z': 10}
    return sum(points[char] for char in word)
```

#1 我们允许将字典值写为多行。

#2 缩进不是必需的，但有助于在视觉上布局字典。

缩进对程序最终执行的结果有很大影响。例如，让我们比较使用连续的两个循环与使用缩进嵌套一个循环在另一个循环中使用的情况。这里有连续的两个循环：

```py
>>> countries = ['Canada', 'USA', 'Japan']
>>> for country in countries:       #1
...     print(country)
...
Canada
USA
Japan
>>> for country in countries:      #2
...     print(country)
...
Canada
USA
Japan
```

#1 这是第一个循环。

#2 这是第二个循环（在第一个循环之后发生）。

这导致我们得到了相同的输出两次，因为我们两次分别遍历了国家列表。现在，如果我们嵌套循环，情况如下：

```py
>>> for country1 in countries:            #1
...     for country2 in countries:        #2
...         print(country1, country2)    #3
...
Canada Canada
Canada USA
Canada Japan
USA Canada
USA USA
USA Japan
Japan Canada
Japan USA
Japan Japan
```

#1 这是第一个循环。

#2 这是第一个循环中的嵌套循环。

#3 `print`是在第二个循环中嵌套的，而第二个循环又嵌套在第一个循环中。

我们为每个`for`循环使用了不同的变量名，`country1`和`country2`，这样我们就可以引用它们。在`country1`循环的第一次迭代中，`country1`指的是`加拿大`。在`country2`循环的第一次迭代中，`country2`同样指的是`加拿大`。这就是为什么第一行输出是`加拿大` `加拿大`。你期望下一行输出是`USA` `USA`吗？但这并不是发生的事情！相反，`country2`循环继续到它的下一次迭代，但`country1`循环还没有移动。`country1`循环只有在`country2`循环完成后才会向前移动。这就是为什么我们在`country1`循环最终移动到第二次迭代之前得到了`加拿大` `USA`和`加拿大` `日本`。当一个循环在另一个循环内部时，这被称为*嵌套循环*。一般来说，当有嵌套时，内循环（`for` `country2` `in` `countries`）将在外循环（`for` `country1` `in` `countries`）移动到它的下一步之前完成所有步骤，然后外循环将重新启动内循环。

如果你看到嵌套在另一个循环内部的循环，那么很可能这些循环正在用于处理二维数据。二维数据组织成行和列，就像你在表格中看到的那样（例如，表 5.1）。这种数据在计算机中非常常见，因为它包括基本的工作表数据，如 CSV 文件，图像如照片或视频的单帧，或者计算机屏幕。

在 Python 中，我们可以使用一个列表来存储二维数据，其中值本身是其他列表。列表中的每个子列表是整体列表中的一行数据，每行都有一个列值。例如，假设我们有关于 2018 年冬季奥运会花样滑冰奖牌的一些数据，如表 5.1 所示。

##### 表 5.1 2018 年冬季奥运会奖牌

| 国家 | 金牌 | 银牌 | 铜牌 |
| --- | --- | --- | --- |
| 加拿大 | 2 | 0 | 2 |
| OAR | 1 | 2 | 0 |
| 日本 | 1 | 1 | 0 |
| 中国 | 0 | 1 | 0 |
| 德国 | 1 | 0 | 0 |

我们可以将这些存储为一个列表，每行一个国家：

```py
>>> medals = [[2, 0, 2],
...           [1, 2, 0],
...           [1, 1, 0],
...           [0, 1, 0],
...           [1, 0, 0]]
```

注意，我们的列表列表只是存储了数值，我们可以通过引用其行和列来找到列表列表中的值（例如，日本的金牌对应于索引为 2 的行和索引为 0 的列）。我们可以使用索引来获取完整的数据行：

```py
>>> medals[0]    #1
[2, 0, 2]
>>> medals[1]    **#2
[1, 2, 0]
>>> medals[-1]    **#3
[1, 0, 0]****
```

****#1 这是第一行（第一行）。

#2 这是第一行（第二行）。

#3 这是最后一行。****  ****如果我们对这个列表执行`for`循环，我们将逐行获取每个完整的行：

```py
>>> for country_medals in medals:     #1
...     print(country_medals)
...
[2, 0, 2]
[1, 2, 0]
[1, 1, 0]
[0, 1, 0]
[1, 0, 0]
```

#1 `for`循环一次给我们列表中的一个值（即一次子列表）。

如果我们只想从奖牌列表中获取特定的值（而不是整个行），我们必须索引两次：

```py
>>> medals[0][0]   #1
2
>>> medals[0][1]    #2
0
>>> medals[1][0]    #3
1
```

#1 这是第一行，第一列。

#2 这是第一行，第一列。

#3 这是第一行，第一列。

假设我们想要逐个遍历每个值。为了做到这一点，我们可以使用嵌套`for`循环。为了帮助我们确切地跟踪我们的位置，我们将使用`range` `for`循环，这样我们就可以打印出当前的行和列数字，以及存储在该处的值。

外层循环将遍历行，因此我们需要使用`range` `(len(medals))`来控制它。内层循环将遍历列。有多少列？嗯，列的数量是行中值的数量，因此我们可以使用`range(len(medals[0]))`来控制这个循环。

每行输出将提供三个数字：行坐标、列坐标以及在该行和列的值（奖牌数量）。以下是代码和输出：

```py
>>> for i in range(len(medals)):          #1
...     for j in range(len(medals[i])):      #2
...             print(i, j, medals[i][j])
...
0 0 2
0 1 0
0 2 2
1 0 1
1 1 2
1 2 0
2 0 1
2 1 1
2 2 0
3 0 0
3 1 1
3 2 0
4 0 1
4 1 0
4 2 0
```

#1 遍历行

#2 遍历当前行的列

注意，在输出的前三行中，行保持不变，而列从 0 到 2 变化。这就是我们如何遍历第一行的方式。只有在行增加到 1 之后，我们才完成对这一新行上列 0 到 2 的工作。

嵌套循环为我们提供了一种系统地遍历二维列表中每个值的系统方法。在处理二维数据时，你经常会看到它们，例如图像、棋盘游戏和电子表格。

### 5.1.3 #8\. 字典

记住，Python 中的每个值都有一个特定的类型。由于我们可能想要使用许多不同类型的值，因此存在许多不同的类型！我们已经讨论了使用数字来处理数值，布尔值来处理`True`/`False`值，字符串来处理文本，以及列表来处理其他值（如数字或字符串）的序列。

在 Python 中，还有一个经常出现的类型，它被称为*字典*。当我们谈论 Python 中的字典时，我们并不是指单词及其定义的列表。在 Python 中，字典是一种在需要跟踪数据之间的关联时非常有用的存储数据的方式。例如，想象一下，如果你想知道你最喜欢的书中使用最频繁的单词。你可以使用字典将每个单词映射到其使用的次数。这样一个字典可能非常大，但这样一个字典的小版本可能看起来像这样：

```py
>>> freq = {'DNA': 11, 'acquire': 11, 'Taxxon': 13, \
... 'Controller': 20, 'morph': 41}
```

字典中的每个条目将一个单词映射到其频率。例如，我们可以从这个字典中得知单词*DNA*出现了 11 次，而单词*Taxxon*出现了 13 次。这里的单词（*DNA*，*acquire*，*Taxxon*等）被称为*键*，而频率（11，11，13 等）被称为*值*。因此，字典将每个键映射到其值。我们不允许有重复的键，但正如这里所示的两个`11`值，有重复的值是没有问题的。

我们在第二章（列表 2.1）中看到了一个字典，它存储了每个四分卫的名字和他们相关的传球码数。在第三章中，我们又看到了一个字典，这是我们的第二个`num_points`解决方案（在列表 5.9 中较早重现）。在那里，字典将每个字母映射到使用该字母所获得的分数。

就像字符串和列表一样，字典也有你可以用来与之交互的方法。以下是一些在`freq`字典上操作的方法：

```py
>>> freq
{'DNA': 11, 'acquire': 11, 'Taxxon': 13, 'Controller': 20, 'morph': 41}
>>> freq.keys()                **#1
dict_keys(['DNA', 'acquire', 'Taxxon', 'Controller', 'morph'])
>>> freq.values()                   #2
dict_values([11, 11, 13, 20, 41])
>>> freq.pop('Controller')         #3
20
>>> freq
{'DNA': 11, 'acquire': 11, 'Taxxon': 13, 'morph': 41}**
```

**#1 获取所有键

#2 获取所有值

#3 删除键及其关联的值**  **你也可以使用索引符号来访问给定键的值：

```py
>>> freq['dna']  # Oops, wrong key name because it is case sensitive
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'dna'
>>> freq['DNA']       #1
11
>>> freq['morph']
41
```

#1 获取与键“DNA”关联的值

字典，就像列表一样，是可变的。这意味着我们可以更改字典中的键和值，这对于模拟随时间变化的数据很有用。我们可以使用索引来更改值。与`'morph'`关联的值目前是`41`。让我们将其更改为`6`：

```py
>>> freq['morph'] = 6     #1
>>> freq
{'DNA': 11, 'acquire': 11, 'Taxxon': 13, 'morph': 6}
```

#1 将与键“morph”关联的值更改为 6

我们的`freq`字典允许我们从任何我们想要的单词开始，并找到它的频率。更普遍地说，字典允许我们从*键到值*进行转换。然而，它不允许我们轻松地朝相反的方向转换，从值到键。如果我们想这样做，我们需要生成相反的字典——例如，一个键是频率，值是具有这些频率的单词列表的字典。这将使我们能够回答以下问题：哪些单词的频率正好是 5？哪些单词的频率是所有单词中的最小或最大？

就像字符串和列表一样，我们也可以使用循环来处理字典中的信息。`for`循环给我们字典的键，我们可以使用索引来获取每个键的关联值：

```py
>>> for word in freq:                         #1
...     print('Word', word, 'has frequency', freq[word])    #2
...
Word DNA has frequency 11
Word acquire has frequency 11
Word Taxxon has frequency 13
Word morph has frequency 6
```

*#1 遍历 freq 字典中的每个键

#2 使用键（单词）和关联值（freq[word])*  *### 5.1.4 #9\. 文件

我们经常需要处理存在于文件中的数据集。例如，在第二章中，我们使用 NFL 统计数据文件来确定最有效的四分卫。使用文件对于其他数据科学任务也很常见。例如，如果你正在绘制关于全球地震的信息或确定两本书是否由同一作者撰写，你需要处理这些数据集，通常这些数据集会存储在文件中。

在第二章中，我们处理了一个名为 nfl_offensive_stats.csv 的文件。请确保这个文件在你的当前程序目录中，因为我们现在将使用这个文件来进一步理解第二章中使用的某些代码。

处理文件中的数据的第一步是使用 Python 的 `open` 函数打开文件：

```py
>>> nfl_file = open('nfl_offensive_stats.csv')
```

你有时会看到 Copilot 在这里添加一个 `r` 作为第二个参数：

```py
>>> nfl_file = open('nfl_offensive_stats.csv', 'r')
```

但我们不需要 `r`；`r` 只意味着我们想要从文件中读取，但如果我们没有指定，这已经是默认的。

我们使用赋值语句将打开的文件分配给名为 `nfl_file` 的变量。现在，我们可以使用 `nfl_file` 来访问文件的内容。一个打开的文件是 Python 类型，就像数字和字符串以及到目前为止你看到的所有其他类型一样。因此，我们可以调用一些方法来与文件交互。其中一个方法是 `readline`，它以字符串的形式给出文件的下一行。我们现在将使用它来获取打开文件的第一行，但不用担心这一行本身，因为它非常长，包含大量我们最终不会使用的列信息：

```py
>>> line = nfl_file.readline()     #1
>>> line
'game_id,player_id,position,player,team,pass_cmp,pass_att,pass_yds,pass_td,pass_int,pass_sacked,pass_sacked_yds,pass_long,pass_rating,rush_att,
rush_yds,rush_td,rush_long,targets,rec,rec_yds,rec_td,rec_long,
fumbles_lost,rush_scrambles,designed_rush_att,comb_pass_rush_play,
comb_pass_play,comb_rush_play,Team_abbrev,Opponent_abbrev,two_point_conv,
total_ret_td,offensive_fumble_recovery_td,pass_yds_bonus,rush_yds_bonus,
rec_yds_bonus,Total_DKP,Off_DKP,Total_FDP,Off_FDP,Total_SDP,Off_SDP,
pass_target_yds,pass_poor_throws,pass_blitzed,pass_hurried,
rush_yds_before_contact,rush_yac,rush_broken_tackles,rec_air_yds,rec_yac,
rec_drops,offense,off_pct,vis_team,home_team,vis_score,home_score,OT,Roof,
Surface,Temperature,Humidity,Wind_Speed,Vegas_Line,Vegas_Favorite,
Over_Under,game_date\n'
```

#1 从文件中读取一行

从这样的混乱字符串中提取单个值并不容易。因此，我们首先倾向于做的是将这样的行拆分成其单个列数据。我们可以使用字符串 `split` 方法来做这件事。该方法接受一个分隔符作为参数，并使用该分隔符将字符串拆分成一个列表：

```py
>>> lst = line.split(',')    #1
>>> len(lst)
69
```

#1 使用逗号 (,) 作为分隔符拆分字符串

现在我们可以查看单个列名：

```py
>>> lst[0]
'game_id'
>>> lst[1]
'player_id'
>>> lst[2]
'position '     #1
>>> lst[3]
'player'
>>> lst[7]
'pass_yds'
```

#1 单词末尾的空格在原始数据集中存在，但其他列标题没有空格。

我们正在查看的文件的第一行不是真实的数据行——它只是告诉我们每个列名的标题。下次我们调用 `readline` 时，我们得到第一行真实的数据：

```py
>>> line = nfl_file.readline()
>>> lst = line.split(',')
>>> lst[3]
'Aaron Rodgers'
>>> lst[7]
'203'
```

逐行移动这种方式适合探索文件中的内容，但最终我们可能想要处理整个文件。要做到这一点，我们可以在文件上使用一个 `for` 循环。它会在每次迭代时给我们一行，我们可以以任何我们喜欢的方式处理它。一旦我们完成了一个文件，我们应该调用它的 `close` 方法：

```py
>>> nfl_file.close()
```

关闭后，我们不再允许使用该文件。现在我们已经讨论了如何读取、处理和关闭文件，让我们看看一个完整的示例。在列表 5.10 中，我们提供了一个第二章程序的版本，该程序按总传球码数对四分卫进行排序。除了展示文件外，我们还使用了第四章和本章中看到的大多数 Python 功能，包括条件语句、字符串、列表、循环和字典。

##### 列表 5.10 不使用 csv 模块的替代 NFL 统计代码

```py
nfl_file = open('nfl_offensive_stats.csv')
passing_yards = {}                    #1

for line in nfl_file:                #2
    lst = line.split(',')
    if lst[2] == 'QB':               #3
        if lst[3] in passing_yards:                   #4
            passing_yards[lst[3]] += int(lst[7])      #5
        else:                                         #6
            passing_yards[lst[3]] = int(lst[7])       #7

nfl_file.close()

for player in sorted(passing_yards, 
                     key=passing_yards.get, 
                     reverse=True):         #8
    print(player, passing_yards[player])
```

#1 这个字典将四分卫的名字映射到他们的传球码数。

#2 遍历文件的每一行

#3 仅关注四分卫

#4 四分卫已经在我们的字典中了。

#5 将四分卫的总数增加；int 将类似'203'的字符串转换为整数。

#6 四分卫尚未在我们的字典中。

#7 设置初始四分卫的总数

#8 从最高到最低传球码数遍历四分卫

底部的这个循环`for player in sorted(passing_yards, key=passing_yards.get, reverse=True):`有很多内容。我们在注释中解释了这一行是按从高到低遍历四分卫。`reverse=True`使我们按从高到低排序，而不是默认的从低到高。`key=passing_yards.get`使排序集中在传球码数（而不是，例如，球员的名字）。如果您想进一步分解这一行代码，请随时向 Copilot 请求进一步解释。这突显了我们在这里试图保持的平衡：知道足够多的知识，能够理解代码的精髓，而不一定需要理解每一个细微之处。

这个程序运行得很好；如果你运行它，你会看到与从第二章运行代码相同的输出。不过，有时候，使用模块（我们将在下一节更深入地介绍模块）可以更容易地编写程序，这就是第二章的程序所做的事情。由于 CSV 文件非常常见，Python 自带了一个模块来简化处理它们。在第二章中，我们给出的解决方案使用了 csv 模块。因此，让我们讨论一下列表 5.10 中的代码（不使用模块）和第二章中的代码（以下列表中重新打印）之间的主要区别（我们给 Copilot 的提示没有显示）。

##### 列表 5.11 使用 csv 模块的 NFL 统计数据代码

```py
# import the csv module
import csv

# open the csv file
with open('nfl_offensive_stats.csv', 'r') as f:    #1
    # read the csv data
    data = list(csv.reader(f))    #2

# create a dictionary to hold the player name and passing yards
passing_yards = {}

# loop through the data
for row in data:                  #3
    # check if the player is a quarterback
    if row[2] == 'QB':
        # check if the player is already in the dictionary
        if row[3] in passing_yards:
            # add the passing yards to the existing value
            passing_yards[row[3]] += int(row[7])
        else:
            # add the player to the dictionary
            passing_yards[row[3]] = int(row[7])

for player in sorted(passing_yards, key=passing_yards.get, reverse=True):
    print(player, passing_yards[player])
```

#1 显示打开文件的另一种语法

#2 使用特殊的 csv 模块；读取文件中的所有数据

#3 遍历每行数据

首先，列表 5.11 使用 csv 模块使处理 CSV 文件变得更容易。csv 模块知道如何操作 CSV 文件，因此，例如，我们不必担心将行拆分成列。其次，列表 5.11 使用了`with`关键字，这意味着当程序完成对该文件的操作时，文件会自动关闭。第三，列表 5.11 在开始任何处理之前先读取整个文件。相比之下，在列表 5.10 中，我们读取并处理每行，一旦读取。

##### 解决编程问题有多种方法

总是存在许多不同的程序可以用来解决同一个任务。有些可能比其他更容易阅读。代码最重要的标准是它能正确地完成工作。之后，我们最关心的是可读性和效率。所以，如果你发现自己难以理解某些代码的工作方式，花些时间查看 Copilot 的其他代码可能值得，以防那里有更简单或更易于理解的解决方案。

在计算任务中，文件被广泛使用，因为它们是常见的数据来源，需要被处理。这包括本节中的 CSV 文件，记录计算机或网站事件日志文件，以及存储你在视频游戏中可能看到的图形数据的文件等。由于文件被如此广泛地使用，因此并不奇怪有许多模块帮助我们读取各种文件格式。这引出了模块的更大主题。

### 5.1.5 #10\. 模块

人们使用 Python 制作各种东西——游戏、网站、用于数据分析、自动化重复任务、控制机器人等应用程序。你可能想知道 Python 怎么可能让你创建如此多种类的程序。当然，Python 的创造者不可能预见到或创建所有需要的支持！

事实上，默认情况下，你的 Python 程序只能访问一些核心 Python 功能（例如我们在上一章和本章中向您展示的那些）。要获取更多功能，我们需要使用模块。而且，要使用一个模块，你需要导入它。

##### Python 中的模块

*模块* 是为特定目的设计的代码集合。回想一下，我们不需要知道一个函数是如何工作的就可以使用它。模块也是一样：我们不需要知道模块是如何工作的就可以使用它们，就像我们不需要知道电灯开关内部是如何工作的就可以使用它一样。作为模块的用户，我们只需要知道模块能帮助我们做什么以及如何编写代码来正确调用其函数。当然，Copilot 可以帮助我们编写这种代码。

当你安装 Python 时，一些模块会随 Python 一起安装，但我们需要导入它们。其他模块我们首先需要安装，然后才能导入。相信我们，如果你想在 Python 中完成特定的任务，可能已经有某人编写了一个模块来帮助你。

你可能想知道如何确定应该使用哪些 Python 模块。你怎么知道哪些模块存在呢？与 Copilot 或 Google 搜索进行简单对话通常很有帮助。例如，如果我们搜索“Python 模块创建 zip 文件”，第一个结果告诉我们所需的模块是 Python 标准库的一部分，这意味着它随 Python 一起提供。如果我们搜索“Python 模块用于可视化”，我们会了解到名为 matplotlib、plotly、seaborn 等模块。搜索这些模块应该会引导你到展示它们功能和典型用途的可视化画廊。大多数模块都可以免费下载和使用，尽管你的搜索结果可以帮助你确认模块是否免费以及其具体的使用许可。我们将在第九章中推迟安装和使用新安装的模块，但到那时，你会看到这个过程：寻找、安装和使用相关模块来帮助我们完成任务。

表 5.2 列出了一些常用的 Python 模块以及它们是否为内置模块。如果一个模块是内置的，你可以直接导入该模块并开始使用它；如果不是，你需要先安装它。

##### 表 5.2 常用 Python 模块总结

| 模块 | 内置 | 描述 |
| --- | --- | --- |
| **#1 创建新的.zip 文件 |
| csv  | 是  | 帮助读取、写入和分析 CSV 文件  |
| zipfile  | 是  | 帮助创建和提取压缩的 zip 存档文件  |
| matplotlib  | 否  | 用于绘图的图形库，作为其他图形库的基础，并提供高度的自定义化  |
| plotly  | 否  | 一个用于创建网络交互式图表的图形库  |
| seaborn  | 否  | 建立在 matplotlib 之上的图形库，可以比 matplotlib 更容易地创建高质量图表  |
| pandas  | 否  | 一个专注于数据框的数据处理库，类似于电子表格  |
| scikit-learn  | 否  | 包含机器学习的基本工具（即，帮助从数据中学习并做出预测）  |
| numpy  | 否  | 提供高效的数据处理  |
| pygame  | 否  | 一个游戏编程库，帮助在 Python 中构建交互式、图形化的游戏  |
| --- | --- | --- |

在第二章中，我们的代码使用了 Python 自带的 csv 模块。让我们继续学习 Python 自带的其他模块。

当人们想要组织他们的文件，可能是在备份或上传之前，他们通常会首先将它们存档到一个.zip 文件中。然后他们可以传递这个单一的.zip 文件，而不是可能成百上千的单独文件。Python 自带了一个名为 zipfile 的模块，可以帮助你创建.zip 文件。

要尝试这个，在你的编程目录中创建一些文件，并让它们都以.csv 结尾。你可以从你的 nfl_offensive_stats.csv 文件开始，然后添加几个更多。例如，你可以添加一个名为 actors.csv 的文件，其中包含一些演员的名字和他们的年龄，如下所示

```py
Actor Name, Age
Anne Hathaway, 40
Daniel Radcliffe, 33
```

你还可以添加一个名为 chores.csv 的文件，其中包含一项任务列表以及你是否完成了每一项：

```py
Chore, Finished?
Clean dishes, Yes
Read Chapter 6, No
```

内容并不重要，只要你有一两个.csv 文件来测试即可。现在我们可以使用 zipfile 模块将它们全部添加到一个新的.zip 文件中！

```py
>>> import zipfile
>>> zf = zipfile.ZipFile('my_stuff.zip', 'w',
    ↪ zipfile.ZIP_DEFLATED)     **#1
>>> zf.write('nfl_offensive_stats.csv')       #2
>>> zf.write('actors.csv')       #3
>>> zf.write('chores.csv')   #4
>>> zf.close()**
```

| django  | 否  | 一个辅助设计网站和 Web 应用的 Web 开发库  |

#2 添加第一个文件

#3 添加第二个文件

#4 添加第三个文件**  **如果你运行这段代码，你会找到一个名为 my_stuff.zip 的新文件，其中包含你的三个.csv 文件。直接使用.zip 文件在以前的其他编程语言中是一个非常专业、容易出错的任务，但 Python 并非如此。Python 自带了一些对数据科学、游戏制作、处理各种文件格式等有帮助的模块，但 Python 并不能提供一切。当我们需要更多的时候，我们会转向可下载的模块，正如我们在第九章中将要看到的。

在本章中，我们向您介绍了我们前 10 个 Python 特性的后半部分，总结如表 5.3 所示。在前一章和本章中，我们讨论了很多关于阅读代码的内容。尽管我们没有涵盖你可能会看到 Copilot 生成的一切，但你处于一个很好的位置来检查 Copilot 代码，以确定它是否尽力按照你请求的方式生成代码。我们还展示了更多使用 Copilot 解释工具的示例，以帮助你理解新代码。在接下来的章节中，我们将看到如何测试 Copilot 生成的代码，以确定其是否正确，以及当它不正确时你可以做什么。

##### 表 5.3 本章 Python 代码特性总结

| 代码元素 | 示例 | 简要描述 |
| --- | --- | --- |
| 循环 | `for` 循环：`for country in countries: print(country)` `while` 循环：`index = 0 while index < 4: print(index) index = index + 1` | 循环允许我们根据需要多次运行相同的代码。当我们知道迭代次数时（例如，字符串中的字符数）使用 `for` 循环，不知道时（例如，要求用户输入强密码）使用 `while` 循环。 |
| 缩进 | `for country in countries: print(country)` | 缩进告诉 Python 何时一段代码属于另一个代码块的一部分（例如，`print` 调用位于 `for` 循环内）。 |
| 字典 | `points = {'a': 1, 'b': 3}` | 字典允许我们将键与值关联起来。例如，键 `'a'` 与值 `1` 相关联。 |
| 文件 | `file = open('chores.csv') first_line = file.readline()` | 文件包含数据，存储在您的计算机上。Python 可以打开许多类型的文件并读取其内容，允许您处理文件中的数据。 |
| 模块 | `import csv` | 模块是已经存在的库，提供了额外的功能。常用的模块包括 csv、numpy、matplotlib、pandas 和 scikit-learn。一些模块包含在标准的 Python 发行版中；其他模块需要单独安装。 |

## 5.2 练习

1.  回想一下我们在列表 5.3 中查看的 `for` 循环代码，用于打印列表中的动物。与章节中的原始示例相比，这段修改后的代码有何不同？具体来说，它产生了哪些额外的输出？

```py
lst = ['cat', 'dog', 'bird', 'fish']

for index in range(len(lst)):
    print('Got', lst[index])
    if lst[index] == 'bird':
        print('Found the bird!')
    print('Hello,', lst[index])
```

1.  2. 考虑以下 `while` 循环代码，试图重复我们在列表 5.3 中使用 `for` 循环所做的操作。当我们运行代码时，我们会注意到它无限期地运行。你能识别并修复导致它无限期运行的错误吗？

```py
lst = ['cat', 'dog', 'bird', 'fish']

index = 0
while index < len(lst):
    print('Got', lst[index])
    print('Hello,', lst[index])
```

1.  3. 将以下代码行排列成 `while` 循环，打印列表中的每个数字，直到遇到数字 7。注意缩进！

```py
 index += 1
 while index < len(numbers) and numbers[index] != 7:
 index = 0
 numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
 print(numbers[index])
```

1.  4. 想一个现实场景，其中 `while` 循环比 `for` 循环更合适。描述这个场景，并解释为什么 `while` 循环是更好的选择。

1.  5. 修改`get_strong_password`函数（或它调用的`is_strong_password`函数）以提供有关输入密码不够强大的具体反馈。例如，如果密码没有大写字母，则打印“密码必须包含大写字母”，如果它不包含数字，则打印“密码必须至少包含一个数字”。

1.  6. 给定以下`print_quarterbacks`函数，你能将其重写为使用“with”语句来打开和关闭文件吗？为什么关闭文件很重要？

```py
def print_quarterbacks():
    nfl_file = open('nfl_offensive_stats.csv')
    for line in nfl_file:
        lst = line.split(',')
        if lst[2] == 'QB':
            print(f"{lst[3]}: {lst[7]} passing yards")
    nfl_file.close()
```

1.  7. 在这个练习中，我们将进一步练习使用 zipfile 模块创建包含多个 CSV 文件的.zip 文件。按照以下步骤完成任务并回答问题：

    1.  首先，在你的当前目录中创建三个 CSV 文件：

        +   nfl_offensive_stats.csv（你应该已经有了这个文件）

        +   actors.csv 包含以下内容：

            ```py
                          Actor Name, Age
                          Anne Hathaway, 40
                          Daniel Radcliffe, 33

            ```

        +   chores.csv 包含以下内容：

            ```py
                          Chore, Finished?
                          Clean dishes, Yes
                          Read Chapter 6, No

            ```

    1.  使用 Copilot（不要直接像我们在本章中那样输入代码），编写一个 Python 脚本，使用 zipfile 模块将这些三个 CSV 文件添加到名为 my_stuff.zip 的.zip 文件中。

    1.  Copilot 建议的 zipfile 模块提供的其他一些功能有哪些？它们有什么用？

## 摘要

+   循环用于重复执行代码，直到满足所需次数。

+   当我们知道循环将执行多少次迭代时，我们使用`for`循环；当我们不知道循环将执行多少次迭代时，我们使用`while`循环。

+   Python 使用缩进来确定哪些代码行属于同一组。

+   字典是从键（例如，一本书中的单词）到值（例如，它们的频率）的映射。

+   在读取文件之前，我们需要先打开文件。

+   一旦文件打开，我们可以使用方法（例如，readline）或循环来读取其行。

+   一些模块，如 csv 和 zipfile，与 Python 一起提供，可以通过导入它们来使用。

+   其他模块，如 matplotlib，需要先安装，然后才能导入和使用。***************
