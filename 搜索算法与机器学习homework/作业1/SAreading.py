#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/17
# @Author  : github.com/guofei9987

"""退火算法（SA）简介和直观感受
非常经典的机器监督学习代码，由于退火物理过程的不确定性：
分子在同一个温度下依然可能出现异常小内能
导致了在进行学习的时候如果仅仅按照梯度下降算法，可能会陷入局部最小值而并非全局最优解
SA经典的地方在于，他引入了有关于温度T以及新旧解之间差距的一个概率P，在温度很大的时候，P近似于1,这个解就算是没有刚才那么的好也会被接受，直到找到最优解...吗？
假设这个局部最小值在非常低的温度出现，但是在此之前却又有另一个最小值出现，而且已经满足了P的收敛条件，那不就没有办法达到全局最优了吗？
实际上，这个问题的解决正是机器学习正在研究的重点之一，如果只是一直探索最小值，函数将不会收敛而在不断振荡；如果追求收敛，那有可能在遇到局部最小值的时候就错误的认为他是全局最小值
SA算法正是尽力在这两者间取平衡的典范.
比如，在爬山过程中，总会出现各种山峰，但假如我没有这个地方的地图，视野也仅仅在我身边的一小部分，只能判断这座山到底是在上升还是下降（比较前后值是否有变好）。
那你会选择怎么样一个策略，来尽可能保证你登上的山是最高的那一座呢？
于是你尝试随便向周围迈出一步随机的长度，沿着上坡的方向一直爬到当前的最高峰。这是第一步。
这个时候，你发现你有个海拔计。你一看：好低的海拔！于是你想，这大概率不是最高的山峰！（P=exp(-Δf/T)）
你接着向前探索，不断寻找，终于，你找到了这样一座山峰：
他的海拔很高，同时足够陡峭，鹤立鸡群。
你觉得这就是最高的山峰了，你站在山顶，四周都是向下的路。
你对自己说：这里应该就是最高的山了。于是心满意足的回家宣传这个消息：你登上了群山中最高的那一座！
但这座山到底是不是最高的山呢？虽然他很高，但前面已经有仅仅是比他低一点点的山头，只是出于某些心理因素，你认为更高的山还在后头。
那谁知道在这座山峰之后还有没有更高的山呢？（P接近0）（局部最优解与全局最优解）

这就是SA算法的通俗解释。如果用一定的数学语言解释的话：
SA算法最关键的地方只有两个：第一个是“怎么产生新解”，第二个是“如何做到有时候接受更差的解”。
假设当前解为：x_new = x_current ​+ Δx
这里的Δx本质上就是一个随机扰动。
SA并不像梯度下降一样严格沿着“下降最快方向”前进，而是会随机在当前解附近进行探索。温度 T 越高，这种随机扰动可能越大；温度越低，搜索范围就会越来越小，最后慢慢稳定下来。
但是SA真正经典的地方并不是随机搜索，而是它允许“接受更差的解”。假设：Δf=f(x_new)−f(x_current)
如果Δf<0,说明新解更优秀，那么直接接受。但如果：Δf>0说明新解反而更差，SA依然有概率接受它：P=exp(−Δf/T)
这个公式我个人理解其实很像一种“容错机制”。
当温度 T 很高的时候，就算当前找到的解更差，算法依然有比较大的概率接受它，因为高温阶段更偏向于“到处探索”，不希望太早卡死在某个局部最小值附近。
随着温度下降，P会越来越小，算法也会越来越“保守”，逐渐从随机探索转向稳定收敛。
所以SA本质上其实是在做一件很矛盾但又很重要的事情：一方面想尽量寻找更小值;另一方面又不能太贪心，否则容易困在局部最优。
而退火过程中的温度 T，就是用来平衡“探索”和“收敛”的核心参数.
"""
"""
SA.py
│
├── 依赖 base.py
│      └── 提供 SkoBase 基类
│
├── 依赖 mutation.py
│      └── 提供 swap/reverse/transpose 等变异算子
│
└── 被 examples/demo_sa.py 调用
"""
"""
核心变量的变化过程：
变量       含义             初始化              变化规律
T        当前温度           T_max              不断下降
x_current  当前解             x0               被接受的新解替换
best_x   历史最优解         x0                 仅在更优时更新
stay_counter 停滞计数器      0                 长时间不更新则增加
"""
import numpy as np
from .base import SkoBase
from sko.operators import mutation
#导入要用的库

class SimulatedAnnealingBase(SkoBase):
    """
    下面的几个其他的类（包括但不限于边界条件，新的算法）都继承了这个base，说明：
    不同的退火策略共享主循环，只重写扰动策略与降温策略
    """
    """
    DO SA(Simulated Annealing)

    Parameters
    ----------------
    func : function
        The func you want to do optimal
    n_dim : int
        number of variables of func
    x0 : array, shape is n_dim
        initial solution
    T_max :float
        initial temperature
    T_min : float
        end temperature
    L : int
        num of iteration under every temperature（Long of Chain）

    Attributes
    ----------------------


    Examples
    -------------
    See https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_sa.py
    """

    def __init__(self, func, x0, T_max=100, T_min=1e-7, L=300, max_stay_counter=150, **kwargs):
        assert T_max > T_min > 0, 'T_max > T_min > 0'
        #L是一个关键参数，这表示在每个温度下搜索多少次
        self.func = func#状态方程，x怎么映射到y上，用于储存目标函数
        self.T_max = T_max  # initial temperature 最初的温度
        self.T_min = T_min  # end temperature 结束的温度，指定最末值  
        self.L = int(L)  # num of iteration under every temperature（also called Long of Chain）
        # stop if best_y stay unchanged over max_stay_counter times (also called cooldown time)
        self.max_stay_counter = max_stay_counter

        self.n_dim = len(x0)

        self.best_x = np.array(x0)  # initial solution
        self.best_y = self.func(self.best_x) #这两行记录的是当前已知最优解x与y，是历史性的，已经得到过的而不仅仅是现在这一时刻的解
        self.T = self.T_max#模拟物理退火过程的起始阶段，从高温度开始降温
        self.iter_cycle = 0
        self.generation_best_X, self.generation_best_Y = [self.best_x], [self.best_y]
        # history reasons, will be deprecated
        self.best_x_history, self.best_y_history = self.generation_best_X, self.generation_best_Y

    def get_new_x(self, x):
        u = np.random.uniform(-1, 1, size=self.n_dim)#给每个维度一个随机扰动，范围均匀分布在（-1,1）
        x_new = x + 20 * np.sign(u) * self.T * ((1 + 1.0 / self.T) ** np.abs(u) - 1.0)
        return x_new
        #本函数用于获得一个新的x值用于向前推进，这个值与之前的之差距是随机的但是不是很大的，而且保证是正的

    def cool_down(self):
        self.T = self.T * 0.7
        #用于下降温度的函数，本函数关键在于其系数，类似于梯度下降的学习率

    def isclose(self, a, b, rel_tol=1e-09, abs_tol=1e-30):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
        #浮点数由于精度问题不能直接比较大小，需要一个误差容忍范围来相等，这个函数就是做这个的

    def run(self): #真正的核心执行函数，让SA算法取值跑起来的主体
        x_current, y_current = self.best_x, self.best_y#初始化，刚开始的时候当前解就是历史最优解，后面就会变化了
        stay_counter = 0
        while True: #温度循环，每循环一次温度就变化一次
            for i in range(self.L):#在（1,L）中遍历循环，即在一个温度下循环L次
                x_new = self.get_new_x(x_current)#生成新的解，其实就是从当前解附近随便小走一步
                y_new = self.func(x_new)

                """Metropolis准则：
                df表示新解比旧解差多少，<0则直接接受新解，>0则进行下一步计算
                if df < 0 or np.exp(-df / self.T) > np.random.rand():
                这一步便是开头注释中所说的概率P随机接受，np.random.rand()生成[0,1)的随机数字
                可以解释为“即使是更差的解也有概率能接受，这个概率在高温下可能性更大”
                尽可能避免因为局部最优解而停滞
                """
                df = y_new - y_current
                if df < 0 or np.exp(-df / self.T) > np.random.rand():
                    x_current, y_current = x_new, y_new#更新当下变量
                    if y_new < self.best_y:
                        self.best_x, self.best_y = x_new, y_new #如果经过这个过程得到的解更优秀，取他作为历史最优解！

            self.iter_cycle += 1
            self.cool_down()#执行降温程序
            self.generation_best_Y.append(self.best_y)
            self.generation_best_X.append(self.best_x)#将当前温度下的最优解添加到列表里

            # if best_y stay for max_stay_counter times, stop iteration
            if self.isclose(self.best_y_history[-1], self.best_y_history[-2]):
                stay_counter += 1
            else:
                stay_counter = 0#这里的意思是，如果两个温度下的最优解一样的话，那么增加一次没有进步的次数，这个次数达到了阀值，就输出“已经达到最优解了”
                #同时要注意如果继续进步的话，停滞次数要清零

            if self.T < self.T_min:
                stop_code = 'Cooled to final temperature'
                break#这是达到我设定的最小温度值所以停止继续探索
            if stay_counter > self.max_stay_counter:
                stop_code = 'Stay unchanged in the last {stay_counter} iterations'.format(stay_counter=stay_counter)
                break#这是已经停止进步太久了，可以认为是最优解

        return self.best_x, self.best_y

    fit = run#本质上给函数起别名


class SimulatedAnnealingValue(SimulatedAnnealingBase):#继承了上面的类SimulatedAnnealingBase，说明是对上面类的以后一种约束和进阶
    """
    SA on real value function
    """

    def __init__(self, func, x0, T_max=100, T_min=1e-7, L=300, max_stay_counter=150, **kwargs):
        super().__init__(func, x0, T_max, T_min, L, max_stay_counter, **kwargs)
        lb, ub = kwargs.get('lb', None), kwargs.get('ub', None)#相当于lb=None，ub=None，不过这里是可扩展输入参数，不输入也没关系，就是是否需要边界条件的取舍

        if lb is not None and ub is not None:#有边界条件的情况下
            self.has_bounds = True
            self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)#统一边界维度，不管输入是什么都可以统一为先前传入参数的维度数目，给每个维度都加上边界条件
            assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'#输入合法化检查，不允许不同维度的输入
            assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'#上边界必须大于下边界
            self.hop = kwargs.get('hop', self.ub - self.lb)#搜索步长获取
        elif lb is None and ub is None:#没边界条件的情况下
            self.has_bounds = False
            self.hop = kwargs.get('hop', 10)#搜索步长获取，默认为10
        else:
            raise ValueError('input parameter error: lb, ub both exist, or both not exist')
        self.hop = self.hop * np.ones(self.n_dim)#搜索步长作用于每一个维度
        """
        这个类就是对SA算法进行的修正，添加了边界条件和搜索步长的考虑
        """

class SAFast(SimulatedAnnealingValue):#依旧继承
    """
    u ~ Uniform(0, 1, size = d)
    y = sgn(u - 0.5) * T * ((1 + 1/T)**abs(2*u - 1) - 1.0)

    xc = y * (upper - lower)
    x_new = x_old + xc

    c = n * exp(-n * quench)
    T_new = T0 * exp(-c * k**quench)
    """

    def __init__(self, func, x0, T_max=100, T_min=1e-7, L=300, max_stay_counter=150, **kwargs):
        super().__init__(func, x0, T_max, T_min, L, max_stay_counter, **kwargs)
        self.m, self.n, self.quench = kwargs.get('m', 1), kwargs.get('n', 1), kwargs.get('quench', 1)
        self.c = self.m * np.exp(-self.n * self.quench)#一些初始化

    def get_new_x(self, x):
        r = np.random.uniform(-1, 1, size=self.n_dim)#生成随机扰动
        xc = np.sign(r) * self.T * ((1 + 1.0 / self.T) ** np.abs(r) - 1.0)
        """ 
        温度高的时候尽可能扩大扰动范围，温度小的时候尽可能缩小扰动范围
        sign(r)表示往哪里走，self.T表示温度，(1 + 1.0 / self.T) ** np.abs(r) - 1.0)让这个函数一般很近，偶尔范围非常大
        """
        x_new = x + xc * self.hop#生成新的解
        if self.has_bounds:
            return np.clip(x_new, self.lb, self.ub)#检查是否符合边界条件，不符合直接强行贴到边界上
        return x_new

    def cool_down(self):
        self.T = self.T_max * np.exp(-self.c * self.iter_cycle ** self.quench)#让降温更快，比原始SA更快更激进


class SABoltzmann(SimulatedAnnealingValue):
    """
    std = minimum(sqrt(T) * ones(d), (upper - lower) / (3*learn_rate))
    y ~ Normal(0, std, size = d)
    x_new = x_old + learn_rate * y

    T_new = T0 / log(1 + k)
    """

    def __init__(self, func, x0, T_max=100, T_min=1e-7, L=300, max_stay_counter=150, **kwargs):
        super().__init__(func, x0, T_max, T_min, L, max_stay_counter, **kwargs)
        self.learn_rate = kwargs.get('learn_rate', 0.5)#依旧继承和初始化，不过这里多了一个“学习率“，其实就是步长系数

    def get_new_x(self, x):
        a, b = np.sqrt(self.T), self.hop / 3.0 / self.learn_rate#a：温度越高随机扰动越大，b：限制标准差最大值，防止随机步长过大
        std = np.where(a < b, a, b)#等价于min（a，b），但是这是向量版本
        xc = np.random.normal(0, 1.0, size=self.n_dim)#高斯分布，size依旧给每个维度上限制
        x_new = x + xc * std * self.learn_rate#利用高斯分布生成新的解
        """
        高斯分布和前面一个类有什么区别？
        高斯分布更集中，主要研究的是某一点附近的变化，连续搜索，前一个函数则是偏向于更激进的探索策略
        """
        if self.has_bounds:
            return np.clip(x_new, self.lb, self.ub)
        return x_new#同样是处理边界条件

    def cool_down(self):
        self.T = self.T_max / np.log(self.iter_cycle + 1.0)#降温，这里的降温速度非常慢


class SACauchy(SimulatedAnnealingValue):#柯西分布的做法
    """
    u ~ Uniform(-pi/2, pi/2, size=d)
    xc = learn_rate * T * tan(u)
    x_new = x_old + xc

    T_new = T0 / (1 + k)
    """

    def __init__(self, func, x0, T_max=100, T_min=1e-7, L=300, max_stay_counter=150, **kwargs):
        super().__init__(func, x0, T_max, T_min, L, max_stay_counter, **kwargs)
        self.learn_rate = kwargs.get('learn_rate', 0.5)#和高斯分布一样

    def get_new_x(self, x):
        u = np.random.uniform(-np.pi / 2, np.pi / 2, size=self.n_dim)
        xc = self.learn_rate * self.T * np.tan(u)
        """
        柯西分布由于数学原因，在接近两端的数值比较大，所以柯西分布会出现偶尔的一步超级大跨越，更容易跳出局部最优解
        比高斯分布求解更适合SA的具体情况，因为SA经常会遇到局部最优解和全局最优解的取舍问题
        """
        x_new = x + xc
        if self.has_bounds:
            return np.clip(x_new, self.lb, self.ub)
        return x_new#依旧边界条件，但这里的边界条件及其重要，因为正切函数边界上取值可能非常的大

    def cool_down(self):
        self.T = self.T_max / (1 + self.iter_cycle)#线性衰减


# SA_fast is the default
SA = SAFast


class SA_TSP(SimulatedAnnealingBase):
    def cool_down(self):
        self.T = self.T_max / (1 + np.log(1 + self.iter_cycle))

    def get_new_x(self, x):
        x_new = x.copy()
        new_x_strategy = np.random.randint(3)
        if new_x_strategy == 0:
            x_new = mutation.swap(x_new)
        elif new_x_strategy == 1:
            x_new = mutation.reverse(x_new)
        elif new_x_strategy == 2:
            x_new = mutation.transpose(x_new)

        return x_new
    """
    简单说一下吧，旅行商问题我也没有很看懂，不过我个人觉得是不太适合SA算法的
    这里的降温用的是对数降温，非常的慢，因为旅行商问题很容易因为贪心算法以及排列组合的局限性卡在局部最小值
    所以需要保留缓慢的降温速度从而让探索更有可能（同样是接受概率的影响）
    这里用x.copy()是不想改变x输入值
    后面用随机数随机三个探索策略，正是提升tsp的搜索能力
    """
"""
为什么要设计多个SA版本呢？
SA算法很重要的一点在于，他有两个对立面即“速率”和“精确”
三种不同的方案（称SAFast为F，SABoltzmann为B，SACauchy为C）是对这两个方面的平衡
F偏向于激进的探索，更追求速度，收敛更加快，但是由于步长的问题，有可能导致错过细节，导致精确度有问题，直接忽略了很多解，有可能直接跨过最好的解。
所以这里的T下降的更快一些，让F尽可能快的到达低温区域，从而更快的收敛。
B偏向于精细化搜索，步长很短，不容易错过最优解，但也正是他这个步子很短的原因带来了一个问题：
他容易陷入局部最优解出不来了。
所以他的温度下降的很慢。这样，对于一些局部最优解，由于温度下降较慢，所以接受差解的概率在较长时间内不会迅速减小，继续往下进行。
而C则更偏向于一种偶尔的跳远。
它使用的是柯西分布，由于柯西分布具有重尾特性，所以虽然大多数时候步长也不算特别大，但会偶尔出现一次非常大的跨越。
这种做法有点像：平时正常搜索，但偶尔突然跳很远看看。
这样的好处是，它比B更容易跳出局部最优；同时又不像F那样一直保持非常激进的搜索，因此在探索能力和稳定性之间取了一个中间值。
所以C可以理解为：既想保留一定的精细搜索能力，又希望在必要的时候拥有“强行跳出局部最优”的能力。
"""


"""
这篇SA算法核心思想就是利用调控温度下降的速度和“随机凭借概率接受可能没那么好的解“来跳出方法函数的局部最小值
学习这篇算法很大程度上帮助我了解机器学习的本质：
机器学习需要在效率和精确度上找平衡，有的时候下降的过快会导致没有找到最好的解，最拟合的解就结束了
有的时候时间复杂度太高，完全不可接受
"""