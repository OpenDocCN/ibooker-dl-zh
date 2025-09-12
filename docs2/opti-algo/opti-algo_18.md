# 附录 B. 基准和数据集

本附录提供了优化基本资源的概述，包括测试函数、组合优化数据集、地理空间数据和机器学习数据集。

## B.1 优化测试函数

*优化测试函数*，也称为*基准函数*，是用于评估优化算法性能的数学函数。以下是一些这些测试函数的例子：

+   *Ackley*——这是一个广泛用于测试优化算法的函数。在其二维形式中，它以几乎平坦的外围区域和中心的大洞为特征。

+   *Bohachevsky*——这是一个具有碗形形状的二维单峰函数。这个函数已知是连续的、凸的、可分离的、可微的、非多模态的、非随机的和非参数的，因此基于导数的求解器可以有效地处理它。请注意，变量可以分离的函数被称为*可分离函数*。*非随机函数*不包含随机变量。*非参数函数*假设数据分布不能通过有限参数集来定义。

+   *Bukin*——这个函数有许多局部最小值，所有这些最小值都位于脊上，并且在 *x*[*] = *f*(–10,1) 处有一个全局最小值 *f*(*x*[*]) = 0。这个函数是连续的、凸的、不可分离的、不可微的、多模态的、非随机的和非参数的。这需要使用无导数求解器（也称为黑盒求解器）如模拟退火。

+   *Gramacy & Lee*——这是一个具有多个局部最小值以及局部和全局趋势的一维函数。这个函数是连续的、非凸的、可分离的、可微的、非多模态的、非随机的和非参数的。

+   *Griewank 1D、2D 和 3D 函数*——这些函数有许多广泛存在的局部最小值。这些函数是连续的、非凸的、可分离的、可微的、多模态的、非随机的和非参数的。

[“B.1_Optimization_test_functions.ipynb”列表](https://github.com/Optimization-Algorithms-Book/Code-Listings/blob/main/Appendix%20B/Listing%20B.1_Optimization_test_functions.ipynb) 在本书的 GitHub 仓库中展示了不同测试函数的示例，这些函数可以从零开始实现或从 Python 框架如 DEAP、pymoo 和 PySwarms 中检索。

## B.2 组合优化基准数据集

[“B.2_CO_datasets.ipynb”列表](https://github.com/Optimization-Algorithms-Book/Code-Listings/blob/main/Appendix%20A/Listing%20A.2_Graph_libraries.ipynb) 在本书的 GitHub 仓库中提供了组合优化问题的基准数据集示例，例如以下这些：

+   *旅行商问题 (TSP)*——给定一组 *n* 个节点以及每对节点之间的距离，找到一次访问每个节点且总长度最小的环形旅行。基准数据集可在 [`github.com/coin-or/jorlib/tree/master/jorlib-core/src/test/resources/tspLib/tsp`](https://github.com/coin-or/jorlib/tree/master/jorlib-core/src/test/resources/tspLib/tsp) 找到。

+   *车辆路径问题（VRP）*—确定一支车队为服务一组客户或位置的最优路线和调度。基准数据集可在[`github.com/coin-or/jorlib/tree/master/jorlib-core/src/test/resources/tspLib/vrp`](https://github.com/coin-or/jorlib/tree/master/jorlib-core/src/test/resources/tspLib/vrp)和[`neumann.hec.ca/chairedistributique/data/`](http://neumann.hec.ca/chairedistributique/data/)找到。

+   *车间作业调度（JSS）*—JSS 涉及在一组机器上对一组作业进行调度，其中每个作业由多个必须在特定顺序上不同机器上处理的操作组成。目标是确定一个最优调度方案，以最小化所有作业的完工时间或总完成时间。基准数据集可在[`people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop1.txt`](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop1.txt)和[`people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop2.txt`](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop2.txt)找到。

+   *装配线平衡问题（ALBP）*—ALBP 涉及将任务（工作元素）分配到工作站，以最小化生产线上的空闲时间，同时满足特定约束。ALBP 通常包括在开始实际装配过程之前与给定生产过程的生产单元装备和调整相关的所有任务和决策。这包括设置系统产能，包括周期时间、工作站数量和工作站设备，以及将工作内容分配给生产单元，包括任务分配和确定操作顺序。基准数据集可在[`assembly-line-balancing.de/`](https://assembly-line-balancing.de/)找到。

+   *二次分配问题（QAP）*—QAP 涉及确定一组设施到一组位置的优化分配。它在运筹学中得到广泛研究，并在设施布局设计、制造、物流和电信等各个领域有应用。基准数据集可在[`mistic.heig-vd.ch/taillard/problemes.dir/qap.dir/qap.html`](http://mistic.heig-vd.ch/taillard/problemes.dir/qap.dir/qap.html)找到。

+   背包问题—给定一个包含*n*个物品的集合，每个物品*i*有一个重量*w[*i*]和一个价值*v[*i*]。你想要选择这些物品的子集，使得所选物品的总重量小于或等于给定的重量限制*W*，并且所选物品的总价值尽可能大。基准数据集可在[`people.brunel.ac.uk/~mastjjb/jeb/orlib/mknapinfo.html`](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/mknapinfo.html)找到。

+   *集合覆盖问题（SCP）*—给定一个包含 *n* 个元素的集合 *U* 和一个包含 *m* 个集合的集合 *S*，其并集等于集合 *U*，集合覆盖问题是要找到 *S* 的最小子集，使得这个子集仍然覆盖集合 *U* 中的所有元素。基准数据集可在[`people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html`](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html)找到。

+   桶装问题—给定一个包含 *n* 个项目的集合，每个项目的大小为 *s*[*i*]，以及一个容量为 *C* 的桶，问题是将每个项目分配到一个桶中，使得每个桶中项目的总大小不超过 *C*，并且使用的桶的数量最小化。基准数据集可在[`people.brunel.ac.uk/~mastjjb/jeb/orlib/binpackinfo.html`](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/binpackinfo.html)找到。

## B.3 地理空间数据集

空间数据是指任何直接或间接引用特定地理位置的数据。此类数据的例子包括但不限于

+   人员、企业、资产、自然资源、新发展、服务和其他建筑基础设施的位置

+   空间分布变量，如交通、健康统计数据、人口统计和天气

+   与环境变化相关的数据—生态、海平面上升、污染、温度等

+   与协调应对紧急情况和自然灾害以及人为灾害的反应相关的数据—洪水、流行病、恐怖主义

如果你的优化问题包括地理空间数据，你可以从多个在线资源和开放数据仓库中检索这些数据。书中 GitHub 仓库的列表 B.3 ([“Listing B.3_Geospatial_data.ipynb”](https://github.com/Optimization-Algorithms-Book/Code-Listings/blob/main/Appendix%20B/Listing%20B.3_Geospatial_data.ipynb)) 和 B.4 ([“Listing B.4_Geospatial_data_TBS.ipynb”](https://github.com/Optimization-Algorithms-Book/Code-Listings/blob/main/Appendix%20B/Listing%20B.4_Geospatial_data_TBS.ipynb)) 展示了如何从开源数据源（如 OpenStreetMap (OSM)、Overpass API、Open-Elevation API 等）获取数据的示例。

## B.4 机器学习数据集

神经组合优化已在各种数据集上得到应用。然而，由于这个领域通常关注解决经典的优化问题，基准数据集通常是这些问题的标准实例。除了列表 B.2 中包含的数据集（[“Listing B.2_CO_datasets.ipynb”](https://github.com/Optimization-Algorithms-Book/Code-Listings/blob/main/Appendix%20B/Listing%20B.2_CO_datasets.ipynb)），书中 GitHub 仓库的列表 B.5 ([“Listing B.5_ML_datasets.ipynb”](https://github.com/Optimization-Algorithms-Book/Code-Listings/blob/main/Appendix%20B/Listing%20B.5_ML_datasets.ipynb)）提供了神经组合优化数据集的示例，例如这些：

+   *凸包*—给定一组点，寻找或计算凸包的问题。凸包是一个几何形状，具体来说是一个多边形，它完全包围了给定的一组点。它通过优化两个不同的参数来实现这一点：它最大化形状覆盖的面积，同时最小化形状的边界或周长。数据可在本书的 GitHub 仓库中找到（在附录 B 数据文件夹中）。

+   *TSP*—用于训练和测试 TSP 的指针网络数据集。数据可在本书的 GitHub 仓库中找到（在附录 B 数据文件夹中）。

+   *TLC 行程记录数据*—黄色和绿色出租车行程记录包括捕获接车和下车日期和时间、接车和下车地点、行程距离、详细费用、费率类型、支付类型和驾驶员报告的乘客计数字段。数据可在[www.nyc.gov/site/tlc/about/tlc-trip-record-data.page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)找到。

## B.5 数据文件夹

书籍 GitHub 仓库中包含的[data](https://github.com/Optimization-Algorithms-Book/Code-Listings/tree/main/Appendix%20B/data)文件夹（[`github.com/Optimization-Algorithms-Book/Code-Listings/tree/main/Appendix%20B/data`](https://github.com/Optimization-Algorithms-Book/Code-Listings/tree/main/Appendix%20B/data)）包括以下示例数据：

+   *ALBP*—第六章中用于生产线平衡问题的数据集

+   *行政边界*—世界各地不同感兴趣区域的行政边界，以 geoJSON 格式提供，可用于地图可视化

+   *自行车共享*—包含匿名行程数据的多伦多自行车共享（TBS）乘客数据，包括行程开始日期和时间、行程结束日期和时间、行程持续时间、行程开始站点、行程结束站点和用户类型（参见附录 B.4：[附录 B.4_Geospatial_data_TBS.ipynb](https://github.com/Optimization-Algorithms-Book/Code-Listings/blob/main/Appendix%20B/Listing%20B.4_Geospatial_data_TBS.ipynb)）

+   *加拿大交通事故*—来自加拿大统计局 2018 年机动车碰撞数据集，包括每 10 万人死亡人数、每十亿车辆公里死亡人数、每十亿车辆公里受伤人数、每 10 万持牌驾驶员死亡人数和每 10 万持牌驾驶员受伤人数（参见附录 A.2：[附录 A.2_ 图库.ipynb](https://github.com/Optimization-Algorithms-Book/Code-Listings/blob/main/Appendix%20A/Listing%20A.2_Graph_libraries.ipynb)）

+   *安大略省健康*—加拿大安大略省的健康区域（参见附录 A.2：[附录 A.2_ 图库.ipynb](https://github.com/Optimization-Algorithms-Book/Code-Listings/blob/main/Appendix%20A/Listing%20A.2_Graph_libraries.ipynb)）

+   *警察*—多伦多警察局公共服务数据（参见附录 A.2：[附录 A.2_ 图库.ipynb](https://github.com/Optimization-Algorithms-Book/Code-Listings/blob/main/Appendix%20A/Listing%20A.2_Graph_libraries.ipynb)）

+   *政治区划*—第八章中使用到的政治区划数据

+   *PtrNets*—指针网络（第十一章）使用的凸包和 TSP 数据

+   *TSP*—旅行商问题实例
