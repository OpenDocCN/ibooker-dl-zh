## 第十五章\. 对未见数据进行的深度学习：介绍联邦学习

**本章内容**

+   深度学习中的隐私问题

+   联邦学习

+   学习检测垃圾邮件

+   窃取联邦学习

+   安全聚合

+   同态加密

+   同态加密联邦学习

> “朋友不会互相监视；真正的友谊也关乎隐私。”
> 
> *斯蒂芬·金，《海特斯堡之心》（1999 年）*

### 深度学习中的隐私问题

#### 深度学习（及其工具）通常意味着你可以访问你的训练数据

如你所敏锐地意识到的，深度学习作为机器学习的一个子领域，全部都是关于从数据中学习。但通常，被学习的数据极其个人化。最有意义的模型与人类生活中最个人化的信息互动，并告诉我们一些可能难以通过其他方式了解的事情。换句话说，深度学习模型可以研究成千上万人的生活，帮助你更好地理解自己。

深度学习的主要自然资源是训练数据（无论是合成的还是自然的）。没有它，深度学习就无法学习；而且因为最有价值的使用案例通常与最个人化的数据集互动，深度学习往往是公司寻求聚合数据的原因。他们需要它来解决特定的使用案例。

但在 2017 年，谷歌发布了一篇非常激动人心的论文和博客文章，对这次对话产生了重大影响。谷歌提出，我们不需要集中一个数据集来在上面训练模型。公司提出了这个问题：如果我们不能将所有数据带到一处，我们能否将模型带到数据那里？这是一个新的、令人兴奋的机器学习子领域，称为*联邦学习*，这正是本章的主题。

| |
| --- |

如果不是将训练数据集带到一处来训练模型，而是能够将模型带到数据生成的任何地方，会怎么样呢？

| |
| --- |

这种简单的逆转极其重要。首先，这意味着为了参与深度学习供应链，人们实际上不必将他们的数据发送给任何人。在医疗保健、个人管理和其他敏感领域，有价值的模型可以在不要求任何人透露个人信息的情况下进行训练。理论上，人们可以保留对自己个人数据唯一副本的控制权（至少在深度学习方面）。

这种技术将对企业竞争和创业中的深度学习竞争格局产生巨大影响。以前不会（或不能，由于法律原因）共享客户数据的大型企业可能仍然可以从这些数据中获得收入。在有些领域，数据的敏感性和监管约束一直是进步的阻力。医疗保健就是一个例子，数据集通常被严格锁定，使得研究变得困难。

### 联邦学习

#### 你不必访问数据集才能从中学习

联邦学习的前提是许多数据集包含对解决问题有用的信息（例如，在 MRI 中识别癌症），但很难以足够的数量访问这些相关的数据集来训练一个足够强大的深度学习模型。主要担忧是，尽管数据集有足够的信息来训练深度学习模型，但它还包含了一些（可能）与学习任务无关的信息，如果泄露可能会对某人造成潜在伤害。

联邦学习是指模型进入一个安全的环境，学习如何解决问题，而不需要数据移动到任何地方。让我们来看一个例子。

```
import numpy as np
from collections import Counter
import random
import sys
import codecs
np.random.seed(12345)
with codecs.open('spam.txt',"r",encoding='utf-8',errors='ignore') as f: *1*
    raw = f.readlines()

vocab, spam, ham = (set(["<unk>"]), list(), list())
for row in raw:
    spam.append(set(row[:-2].split(" ")))
    for word in spam[-1]:
        vocab.add(word)

with codecs.open(`ham.txt',"r",encoding='utf-8',errors='ignore') as f:
    raw = f.readlines()

for row in raw:
    ham.append(set(row[:-2].split(" ")))
    for word in ham[-1]:
        vocab.add(word)

vocab, w2i = (list(vocab), {})
for i,w in enumerate(vocab):
    w2i[w] = i

def to_indices(input, l=500):
    indices = list()
    for line in input:
        if(len(line) < l):
            line = list(line) + ["<unk>"] * (l - len(line))
            idxs = list()
            for word in line:
                idxs.append(w2i[word])
            indices.append(idxs)
    return indices
```

+   ***1* 数据集来自 [`www2.aueb.gr/users/ion/data/enron-spam/`](http://www2.aueb.gr/users/ion/data/enron-spam/)**

### 学习检测垃圾邮件

#### 假设你想要在人们的电子邮件上训练一个模型来检测垃圾邮件

我们将要讨论的使用案例是电子邮件分类。第一个模型将在一个公开可用的数据集上训练，这个数据集被称为 Enron 数据集，它是一批来自著名 Enron 诉讼案（现在是一个行业标准的电子邮件分析语料库）的大量电子邮件。有趣的事实：我曾经认识一个人，他专业地阅读/注释了这个数据集，人们互相发送了各种各样的疯狂东西（其中很多非常私人）。但由于它在法庭案件中公开发布，现在可以免费使用。

上一节和这一节的代码只是预处理。输入数据文件（ham.txt 和 spam.txt）可在本书的网站上找到，[www.manning.com/books/grokking-deep-learning](http://www.manning.com/books/grokking-deep-learning)；以及 GitHub 上[`github.com/iamtrask/Grokking-Deep-Learning`](https://github.com/iamtrask/Grokking-Deep-Learning)。你预处理它以准备好将其前向传播到在第十三章中创建的嵌入类。和之前一样，这个语料库中的所有单词都被转换成了索引列表。你还通过截断电子邮件或用 `<unk>` 标记填充它，使所有电子邮件正好 500 个单词长。这样做使得最终数据集是方形的。

```
spam_idx = to_indices(spam)
ham_idx = to_indices(ham)

train_spam_idx = spam_idx[0:-1000]
train_ham_idx = ham_idx[0:-1000]

test_spam_idx = spam_idx[-1000:]
test_ham_idx = ham_idx[-1000:]

train_data = list()
train_target = list()

test_data = list()
test_target = list()

for i in range(max(len(train_spam_idx),len(train_ham_idx))):
    train_data.append(train_spam_idx[i%len(train_spam_idx)])
    train_target.append([1])

    train_data.append(train_ham_idx[i%len(train_ham_idx)])
    train_target.append([0])

for i in range(max(len(test_spam_idx),len(test_ham_idx))):
    test_data.append(test_spam_idx[i%len(test_spam_idx)])
    test_target.append([1])

    test_data.append(test_ham_idx[i%len(test_ham_idx)])
    test_target.append([0])

def train(model, input_data, target_data, batch_size=500, iterations=5):
    n_batches = int(len(input_data) / batch_size)
    for iter in range(iterations):
        iter_loss = 0
        for b_i in range(n_batches):

            # padding token should stay at 0
            model.weight.data[w2i['<unk>']] *= 0
            input = Tensor(input_data[b_i*bs:(b_i+1)*bs], autograd=True)
           target = Tensor(target_data[b_i*bs:(b_i+1)*bs], autograd=True)

            pred = model.forward(input).sum(1).sigmoid()
            loss = criterion.forward(pred,target)
            loss.backward()
            optim.step()

            iter_loss += loss.data[0] / bs

            sys.stdout.write("\r\tLoss:" + str(iter_loss / (b_i+1)))
        print()
    return model

def test(model, test_input, test_output):

    model.weight.data[w2i['<unk>']] *= 0

    input = Tensor(test_input, autograd=True)
    target = Tensor(test_output, autograd=True)

    pred = model.forward(input).sum(1).sigmoid()
    return ((pred.data > 0.5) == target.data).mean()
```

有这些不错的 `train()` 和 `test()` 函数，你可以使用以下几行代码初始化一个神经网络并对其进行训练。仅经过三次迭代，网络就可以在测试数据集上以 99.45%的准确率进行分类（测试数据集是平衡的，所以这相当不错）：

|

```
model = Embedding(vocab_size=len(vocab), dim=1)
model.weight.data *= 0
criterion = MSELoss()
optim = SGD(parameters=model.get_parameters(), alpha=0.01)

for i in range(3):
    model = train(model, train_data, train_target, iterations=1)
    print("% Correct on Test Set: " + \
          str(test(model, test_data, test_target)*100))
```

|

|

```
       Loss:0.037140416860871446
% Correct on Test Set: 98.65
       Loss:0.011258669226059114
% Correct on Test Set: 99.15
       Loss:0.008068268387986223
% Correct on Test Set: 99.45
```

|

### 让我们将其变为联邦学习

#### 之前的例子是普通的深度学习。让我们保护隐私

在上一节中，你得到了电子邮件的例子。现在，让我们把所有电子邮件放在一个地方。这是老式的方法（这在世界上仍然非常普遍）。让我们首先模拟一个联邦学习环境，它包含多个不同的电子邮件集合：

```
bob = (train_data[0:1000], train_target[0:1000])
alice = (train_data[1000:2000], train_target[1000:2000])
sue = (train_data[2000:], train_target[2000:])
```

足够简单。现在你可以像以前一样进行相同的训练，但同时在每个人的电子邮件数据库中进行。每次迭代后，你将平均鲍勃、爱丽丝和苏的模型值并评估。请注意，一些联邦学习的聚合方法在每个批次（或批次集合）之后进行；我保持简单：

```
for i in range(3):
    print("Starting Training Round...")
    print("\tStep 1: send the model to Bob")
    bob_model = train(copy.deepcopy(model), bob[0], bob[1], iterations=1)

    print("\n\tStep 2: send the model to Alice")
    alice_model = train(copy.deepcopy(model),
                        alice[0], alice[1], iterations=1)

    print("\n\tStep 3: Send the model to Sue")
    sue_model = train(copy.deepcopy(model), sue[0], sue[1], iterations=1)

    print("\n\tAverage Everyone's New Models")
    model.weight.data = (bob_model.weight.data + \
                         alice_model.weight.data + \
                         sue_model.weight.data)/3

    print("\t% Correct on Test Set: " + \
          str(test(model, test_data, test_target)*100))

    print("\nRepeat!!\n")
```

下一个部分将展示结果。模型的学习效果几乎与之前相同，从理论上讲，你没有访问到训练数据——或者你有吗？毕竟，每个人都在以某种方式改变模型，对吧？你真的不能发现任何关于他们的数据集的信息吗？

```
Starting Training Round...
  Step 1: send the model to Bob
  Loss:0.21908166249699718

           ......

   Step 3: Send the model to Sue
 Loss:0.015368461608470256

 Average Everyone's New Models
 % Correct on Test Set: 98.8
```

### 窃取联邦学习

#### 让我们用一个玩具示例来看看如何仍然学习训练数据集

联邦学习面临两大挑战，这两个挑战在训练数据集中每个人只有少量训练示例时最为严重。这些挑战是性能和隐私。实际上，如果某人只有少量训练示例（或者他们发送给你的模型改进只使用了少量示例：一个训练批次），你仍然可以学到很多关于数据的信息。假设有 10,000 人（每人有一些数据），你将花费大部分时间在来回发送模型，而不是在训练（尤其是如果模型非常大时）。

但我们跑题了。让我们看看当用户在单个批次上执行权重更新时，你能学到什么：

```
import copy

bobs_email = ["my", "computer", "password", "is", "pizza"]

bob_input = np.array([[w2i[x] for x in bobs_email]])
bob_target = np.array([[0]])

model = Embedding(vocab_size=len(vocab), dim=1)
model.weight.data *= 0

bobs_model = train(copy.deepcopy(model),
                   bob_input, bob_target, iterations=1, batch_size=1)
```

鲍勃将使用他收件箱中的一封电子邮件来创建对模型的更新。但鲍勃把他自己的密码保存在一封发给自己的电子邮件中，说：“我的电脑密码是披萨。”愚蠢的鲍勃。通过查看哪些权重发生了变化，你可以推断出鲍勃电子邮件的词汇（并推断其含义）：

```
for i, v in enumerate(bobs_model.weight.data - model.weight.data):
    if(v != 0):
    print(vocab[i])
```

```
is
pizza
computer
password
my
```

就这样，你学会了鲍勃的超秘密密码（也许还有他最喜欢的食物）。怎么办？如果从权重更新中很容易看出训练数据，你该如何使用联邦学习？

### 安全聚合

#### 在任何人看到之前，让我们平均来自成千上万人的权重更新

解决方案是永远不要让鲍勃像那样公开地发布梯度。如果人们不应该看到它，鲍勃如何贡献他的梯度？社会科学使用一种有趣的技巧，称为*随机响应*。

它是这样的。假设你正在进行一项调查，你想询问 100 个人他们是否犯过严重的罪行。当然，即使你承诺不会告诉任何人，他们也会回答“没有”。相反，你让他们抛两次硬币（在你看不见的地方），并告诉他们如果第一次抛硬币是正面，他们应该诚实地回答；如果是反面，他们应该根据第二次抛硬币的结果回答“是”或“否”。

在这种情况下，你实际上从未要求人们告诉你他们是否犯了罪。真正的答案隐藏在第一次和第二次抛硬币的随机噪声中。如果 60%的人说“是”，你可以通过简单的数学计算确定，大约 70%的受访者犯了严重的罪行（上下几个百分点）。这个想法是，随机噪声使得你了解到关于个人的任何信息可能来自噪声而不是他们自己。

| |
| --- |

**通过可辩驳的否认来保护隐私**

特定答案来自随机噪声而不是个人的概率水平，通过提供可辩驳的否认来保护他们的隐私。这构成了安全聚合的基础，以及更广泛地，差分隐私的大部分内容。

| |
| --- |

你只看到整体的汇总统计数据。（你永远不会直接看到任何人的答案；你只看到答案对或更大的分组。）因此，在添加噪声之前，你可以聚合更多的人，你就不需要添加太多的噪声来隐藏他们（并且结果会更加准确）。

在联邦学习的背景下，你可以（如果你愿意）添加大量的噪声，但这会损害训练。相反，首先将所有参与者的梯度求和，这样没有人能看到除了自己的梯度以外的任何人的梯度。这类问题被称为 *安全聚合*，为了做到这一点，你还需要一个额外的（非常酷）工具：*同态加密*。

### 同态加密

#### 你可以对加密值执行算术运算

研究中最激动人心的前沿之一是人工智能（包括深度学习）与密码学的交叉领域。在这个激动人心的交叉点中，有一个非常酷的技术叫做同态加密。简单来说，同态加密允许你在不解密的情况下对加密值进行计算。

尤其是我们对在这些值上执行加法感兴趣。详细解释其工作原理需要一本整本书，但我会用几个定义来展示它是如何工作的。首先，一个 *公钥* 允许你加密数字。一个 *私钥* 允许你解密加密的数字。加密的值称为 *密文*，未加密的值称为 *明文*。

让我们通过使用 phe 库的例子来看看同态加密。（要安装库，请运行 `pip install phe` 或从 GitHub [`github.com/n1analytics/python-paillier`](https://github.com/n1analytics/python-paillier) 下载）：

```
import phe

public_key, private_key = phe.generate_paillier_keypair(n_length=1024)

x = public_key.encrypt(5)         *1*

y = public_key.encrypt(3)         *2*

z = x + y                         *3*

z_ = private_key.decrypt(z)       *4*
print("The Answer: " + str(z_))
```

+   ***1* 加密数字 5**

+   ***2* 加密数字 3**

+   ***3* 将两个加密值相加**

+   ***4* 解密结果**

```
The Answer: 8
```

这段代码在加密状态下将两个数字（5 和 3）相加。非常巧妙，不是吗？还有一种技术与同态加密有点类似：*安全多方计算*。你可以在“密码学与机器学习”博客（[`mortendahl.github.io`](https://mortendahl.github.io)）上了解它。

现在，让我们回到安全聚合的问题。鉴于你新获得的知识，你可以将你看不见的数字相加，答案就变得显而易见了。初始化模型的个人将一个`public_key`发送给鲍勃、爱丽丝和苏，这样他们就可以分别加密他们的权重更新。然后，鲍勃、爱丽丝和苏（他们没有私钥）直接相互沟通，并将所有梯度累积成一个单一、最终的更新，发送回模型所有者，该所有者使用`private_key`对其进行解密。

### 同态加密联邦学习

#### 让我们使用同态加密来保护正在聚合的梯度

```
model = Embedding(vocab_size=len(vocab), dim=1)
model.weight.data *= 0

# note that in production the n_length should be at least 1024
public_key, private_key = phe.generate_paillier_keypair(n_length=128)

def train_and_encrypt(model, input, target, pubkey):
    new_model = train(copy.deepcopy(model), input, target, iterations=1)

    encrypted_weights = list()
    for val in new_model.weight.data[:,0]:
        encrypted_weights.append(public_key.encrypt(val))
    ew = np.array(encrypted_weights).reshape(new_model.weight.data.shape)

    return ew

for i in range(3):
    print("\nStarting Training Round...")
    print("\tStep 1: send the model to Bob")
    bob_encrypted_model = train_and_encrypt(copy.deepcopy(model),
                                            bob[0], bob[1], public_key)

    print("\n\tStep 2: send the model to Alice")
    alice_encrypted_model=train_and_encrypt(copy.deepcopy(model),
                                            alice[0],alice[1],public_key)

    print("\n\tStep 3: Send the model to Sue")
    sue_encrypted_model = train_and_encrypt(copy.deepcopy(model),
                                            sue[0], sue[1], public_key)

    print("\n\tStep 4: Bob, Alice, and Sue send their")
    print("\tencrypted models to each other.")
    aggregated_model = bob_encrypted_model + \
                       alice_encrypted_model + \
                       sue_encrypted_model

    print("\n\tStep 5: only the aggregated model")
    print("\tis sent back to the model owner who")
    print("\t can decrypt it.")
    raw_values = list()
    for val in sue_encrypted_model.flatten():
        raw_values.append(private_key.decrypt(val))
    new = np.array(raw_values).reshape(model.weight.data.shape)/3
    model.weight.data = new

    print("\t% Correct on Test Set: " + \
              str(test(model, test_data, test_target)*100))
```

现在，你可以运行新的训练方案，它增加了一个步骤。爱丽丝、鲍勃和苏在将模型发送回你之前，将他们的同态加密模型相加，这样你就永远不会看到哪些更新来自哪个人（一种合理的否认形式）。在生产中，你还会添加一些额外的随机噪声，足以满足鲍勃、爱丽丝和苏（根据他们的个人偏好）所需的一定隐私阈值。更多内容将在未来的工作中介绍。

```
Starting Training Round...
  Step 1: send the model to Bob
  Loss:0.21908166249699718

  Step 2: send the model to Alice
  Loss:0.2937106899184867

             ...
             ...
             ...

  % Correct on Test Set: 99.15
```

### 摘要

#### 联邦学习是深度学习中最激动人心的突破之一

我坚信，联邦学习将在未来几年改变深度学习的格局。它将解锁之前由于过于敏感而无法处理的新的数据集，从而为这种新出现的创业机会创造巨大的社会效益。这是加密与人工智能研究更广泛融合的一部分，在我看来，这是十年中最激动人心的融合。

阻碍这些技术在实际应用中发挥作用的因素主要是它们在现代深度学习工具包中的不可用性。转折点将是任何人都可以运行`pip install...`然后获得访问深度学习框架的权限，在这些框架中，隐私和安全是首要公民，并且内置了联邦学习、同态加密、差分隐私和安全多方计算等技术（而且你不需要是专家就能使用它们）。

出于这种信念，我在过去一年中作为 OpenMined 项目的一部分，与一群开源志愿者一起工作，将这些原语扩展到主要的深度学习框架中。如果你相信这些工具对未来隐私和安全的重要性，请访问我们的网站[`openmined.org`](http://openmined.org)或 GitHub 仓库([`github.com/OpenMined`](https://github.com/OpenMined))。即使只是给几个仓库点个赞，也请表达你的支持；如果你能加入我们，那就更好了（聊天室：slack.openmined.org）。
