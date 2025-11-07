## 附录 C. 围棋程序和服务器

本附录涵盖了离线和在线玩围棋的各种方法。首先，我们将向您展示如何在本地上安装并使用两个围棋程序——GNU Go 和 Pachi 进行对弈。其次，我们将向您推荐几个流行的围棋服务器，您可以在这些服务器上找到各种实力的人类和人工智能对手。

### 围棋程序

让我们从在您的计算机上安装围棋程序开始。我们将向您介绍两个经典、免费的程序，这些程序已经存在多年。*GNU Go* 和 *Pachi* 都使用了我们在第四章（kindle_split_016.xhtml#ch04）中部分介绍的经典游戏人工智能方法。我们介绍这些工具不是为了讨论它们的方法，而是为了有两个本地测试用的对手，并且可以与它们进行娱乐性的对弈。

与大多数其他围棋程序一样，Pachi 和 GNU Go 可以使用我们在第八章（kindle_split_020.xhtml#ch08）中介绍的围棋文本协议 (GTP)。这两个程序可以以不同的方式运行，这对我们来说非常有用：

+   您可以从命令行运行它们，并通过交换 GTP 命令来玩游戏。这种模式就是您在第八章（kindle_split_020.xhtml#ch08）中使用的，让您的机器人与 GNU Go 和 Pachi 进行对弈。

+   这两个程序都可以安装并使用 *GTP 前端*，这是一种图形用户界面，使得作为人类玩家玩这些围棋引擎变得更加有趣。

#### GNU Go

GNU Go 于 1989 年开发，是现存最古老的围棋引擎之一。最新的发布版本是在 2009 年。尽管最近的发展不多，GNU Go 仍然在许多围棋服务器上作为初学者的流行人工智能对手。此外，它也是基于手工规则的最强大的围棋引擎之一；这为 MCTS 和深度学习机器人提供了一个很好的对比。您可以从 [www.gnu.org/software/gnugo/download.html](http://www.gnu.org/software/gnugo/download.html) 下载并安装 GNU Go，适用于 Windows、Linux 和 macOS。此页面包括安装 GNU Go 作为命令行界面 (CLI) 工具的说明以及链接到各种图形界面。要安装 CLI 工具，您需要从 [`ftp.gnu.org/gnu/gnugo/`](http://ftp.gnu.org/gnu/gnugo/) 下载最新的 GNU Go 二进制文件，解压相应的 tarball，并遵循下载中包含的 INSTALL 和 README 文件中针对您平台的要求。对于图形界面，我们推荐从 [www.rene-grothmann.de/jago/](http://www.rene-grothmann.de/jago/) 安装适用于 Windows 和 Linux 的 JagoClient，以及从 [`sente.ch/software/goban/freegoban.html`](http://sente.ch/software/goban/freegoban.html) 安装适用于 macOS 的 FreeGoban。为了测试您的安装，您可以运行以下命令：

```
gnugo --mode gtp
```

这将启动 GNU Go 的 GTP 模式。程序将在 19×19 棋盘上启动新游戏，并接受来自命令行的输入。例如，您可以通过输入`genmove white`并按 Enter 键来要求 GNU Go 生成一个白子走法。这将返回一个`=`符号以表示有效命令，后跟走法的坐标。例如，响应可能是`= C3`。在第八章中，您将使用 GNU Go 的 GTP 模式作为您自己的深度学习机器人的对手。

当您选择安装图形界面时，您可以立即开始与 GNU Go 对弈并测试您的技能。

#### Pachi

您可以在[`pachi.or.cz/`](http://pachi.or.cz/)下载 Pachi，这是一个整体上比 GNU Go 强大的程序。同时，Pachi 的源代码和详细的安装说明可以在 GitHub 上找到，网址为[`github.com/pasky/pachi`](https://github.com/pasky/pachi)。要测试 Pachi，请在命令行中运行`pachi`，然后输入`genmove black`以让它为您在 9×9 棋盘上生成一个黑子走法。

### 围棋服务器

在您的电脑上与围棋程序对弈可以很有趣且有用，但在线围棋服务器提供了更丰富、更强的对手库，包括人类和人工智能。人类和机器人可以在这些平台上注册账户并参加排名比赛，以提高他们的游戏水平和最终评级。对于人类玩家来说，这提供了一个更具竞争性和互动性的竞技场，以及让您的机器人面对全球玩家的终极考验。您可以在 Sensei 的图书馆中找到详尽的围棋服务器列表，[`senseis.xmp.net/?GoServers`](https://senseis.xmp.net/?GoServers)。在这里，我们展示了三个带有*英文客户端*的服务器。这是一个有偏见的列表，因为迄今为止，最大的围棋服务器是中文、韩文或日文，并且没有提供英文语言支持。由于本书是用英文编写的，我们希望您能够访问到可以在此语言中导航的围棋服务器。

#### OGS

*在线围棋服务器*（OGS）是一个设计精美的基于网络的围棋平台，您可以在[`online-go.com/`](https://online-go.com/)找到它。OGS 是我们用来演示如何在第八章和附录 E 中连接机器人的围棋服务器。OGS 功能丰富，更新频繁，拥有活跃的管理员团队，并且在西半球是最受欢迎的围棋服务器之一。除此之外，我们非常喜欢它。

#### IGS

*互联网围棋服务器*（IGS），可在[`pandanet-igs.com/communities/pandanet`](http://pandanet-igs.com/communities/pandanet)找到，创建于 1992 年，是现有围棋服务器中最古老的之一。它继续受到欢迎，并在 2013 年获得了新界面的翻新。它是少数几个具有原生 Mac 客户端的围棋服务器之一。IGS 是更具竞争力的围棋服务器之一，拥有全球用户基础。

#### Tygem

位于韩国的*Tygem*可能是这里所介绍的三款围棋服务器中用户基础最广泛的；无论何时登录，你都会发现成千上万的玩家，他们处于各个水平。它也非常具有竞争性。世界上许多最强大的围棋专业选手都在 Tygem 上玩（有时是匿名地）。你可以在[www.tygemgo.com](http://www.tygemgo.com)找到它。
