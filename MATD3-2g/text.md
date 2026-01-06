联合解耦用户关联与轨迹设计问题建模过程
系统模型与问题建模
网络模型
考虑一个异构网络，包含一个宏基站（MBS）和 $M$ 个无人机（UAV）作为空中基站。用户设备（UE）集合为 $\mathcal{N} = {1, 2, \dots, N}$，基站集合为 $\mathcal{M} = {0, 1, \dots, M}$，其中索引 $0$ 表示 MBS。

UAV 在固定高度 $h$ 飞行。UE $n$ 在时隙 $t$ 的位置为：

$\mathbf{l}_n(t) = (x_n(t), y_n(t))$

UAV $m$ 在时隙 $t$ 的位置为：

$\mathbf{l}_m(t) = (x_m(t), y_m(t), h(t))$

UE $n$ 与 UAV $m$ 之间的欧氏距离为：

$d_{mn}(t) = \sqrt{(x_m(t) - x_n(t))^2 + (y_m(t) - y_n(t))^2 + h(t)^2}$

## 信道模型
### LoS 概率

$P^{\text{LoS}}_{mn}(t) = \left[ 1 + c_1 \exp\left(-c_2 \left( \vartheta_{mn}(t) - c_1 \right) \right) \right]^{-1}$

其中仰角为：  

$\vartheta_{mn}(t) = \frac{180}{\pi} \arcsin\left( \frac{h(t)}{d_{mn}(t)} \right)$

### NLoS 概率为：

$P^{\text{NLoS}}_{mn}(t) = 1 - P^{\text{LoS}}_{mn}(t)$

## 路径损耗
### LoS 路径损耗：

$PL^{\text{LoS}}_{mn}(t) = \alpha_{mn}(t) \beta^{\text{LoS}}$

### NLoS 路径损耗：

$PL^{\text{NLoS}}_{mn}(t) = \alpha_{mn}(t) \beta^{\text{NLoS}}$

其中：

$\alpha_{mn}(t) = \left( \frac{4\pi f_c d_{mn}(t)}{c} \right)^2$

平均路径损耗：

$PL_{mn}(t) = P^{\text{LoS}}_{mn}(t) PL^{\text{LoS}}_{mn}(t) + P^{\text{NLoS}}_{mn}(t) PL^{\text{NLoS}}_{mn}(t)$

## 关联与干扰模型
### 关联变量定义
定义二进制关联变量：

$b^{(i)}_{mn}(t) =
\begin{cases}
1, & \text{若基站 } m \text{ 与 UE } n \text{ 在 } i \text{ 方向关联} \\
0, & \text{否则}
\end{cases}$

其中 $i \in {U, D}$，分别表示上行（UL）和下行（DL）。

关联约束：

$\sum_{m \in \mathcal{M}} b^{(U)}_{mn}(t) = 1, \quad \sum_{m \in \mathcal{M}} b^{(D)}_{mn}(t) = 1, \quad \forall n \in \mathcal{N}, \forall t$

### 干扰模型（全双工 FD 模式）
对于耦合关联（DUCo）：

UL 基站 $m$ 处的干扰：

$I^{(U)}_m(t) = I^{(U)}_{m,U}(t) + I^{(D)}_{m,U}(t) + I^{\text{self}}_m$

其中：


$I^{(U)}_{m,U}(t) = \sum_{\substack{n' \in \mathcal{N} \\ n' \neq n}} b^{(U)}_{mn'}(t) p_{n'}(t) \cdot PL_{mn'}(t) $

$I^{(D)}_{m,U}(t) = \sum_{\substack{m' \in \mathcal{M} \\ m' \neq m}} b^{(D)}_{m'n}(t) p_{m'}(t) \cdot PL_{m'm}(t) $

$I^{\text{self}}_m = \frac{p_m(t)}{\xi}$

对于解耦关联（DUDe）：

假设 UE $n$ 在 DL 与基站 $v$ 关联，在 UL 与基站 $u$ 关联（$v, u \in \mathcal{M}$）。

UL 基站 $u$ 处的干扰：

$I^{(U)}_u(t) = I^{(D)}_{u,u}(t) + I^{(U)}_{u,u}(t) + I^{\text{self}}_u$

其中：

$I^{(D)}_{u,u}(t) = \sum_{\substack{v \in \mathcal{M} \\ v \neq u}} b^{(D)}_{vn}(t) p_v(t) \cdot PL_{vu}(t)$ 

$I^{(U)}_{u,u}(t) = \sum_{\substack{n' \in \mathcal{N} \\ n' \neq n}} b^{(U)}_{un'}(t) p_{n'}(t) \cdot PL_{un'}(t)$

## 传输速率模型
### UE 与基站间的传输速率：

$\Phi^{(i)}_{mn}(t) = b^{(i)}_{mn}(t) B \log_2\left( 1 + \text{SINR}^{(i)}_{mn}(t) \right)$

其中：

$\text{SINR}^{(U)}_{mn}(t) = \frac{p_n(t) PL_{mn}(t)}{I^{(U)}_m(t) + \sigma^2}$

$\text{SINR}^{(D)}_{mn}(t) = \frac{p_m(t) PL_{mn}(t)}{I^{(D)}_n(t) + \sigma^2}$

回程速率（UAV 与 MBS 之间）：

$\Phi^{\text{back}}_{m}(t) = B^{\text{back}} \log_2\left( 1 + \frac{p_0(t) PL_{m0}(t)}{\sigma^2} \right)$

## 优化问题建模
### 决策变量
关联变量：$\mathbf{b} = { b^{(U)}{mn}(t), b^{(D)}{mn}(t) }$

UAV 轨迹：$\mathbf{L} = { \mathbf{l}m(t) }{m \in \mathcal{M}\backslash{0}}$

功率分配：$\mathbf{p} = { p_m(t), p_n(t) }$

## 目标函数
### 最大化总时间内所有 UE 在 UL 和 DL 的平均总速率：

$\max_{\mathbf{b},\mathbf{L},\mathbf{p}} \frac{1}{T} \left( \sum_{t=1}^{T} \sum_{m \in \mathcal{M}} \sum_{n \in \mathcal{N}} \Phi^{(U)}_{mn}(t) + \sum_{t=1}^{T} \sum_{m \in \mathcal{M}} \sum_{n \in \mathcal{N}} \Phi^{(D)}_{mn}(t) \right)$

## 约束条件
关联唯一性约束：

$\sum_{m \in \mathcal{M}} b^{(U)}_{mn}(t) = 1, \quad \sum_{m \in \mathcal{M}} b^{(D)}_{mn}(t) = 1, \quad \forall n \in \mathcal{N}, \forall t$

回程容量约束：

$\sum_{n \in \mathcal{N}} \Phi^{(U)}_{mn}(t) + \sum_{n \in \mathcal{N}} \Phi^{(D)}_{mn}(t) \leq \Phi^{\text{back}}_{m}(t), \quad m \in \mathcal{M}\backslash\{0\}, \forall t$

UAV间安全距离约束：

$\| \mathbf{l}_m(t) - \mathbf{l}_{m'}(t) \|_2 \geq d_{\min}, \quad m \neq m' \in \mathcal{M}\backslash\{0\}, \forall t$

UAV最大速度约束：

$\| \mathbf{l}_m(t+1) - \mathbf{l}_m(t) \|_2 \leq v_{\max} \Delta t, \quad m \in \mathcal{M}\backslash\{0\}, \forall t$

轨迹闭合约束：

$\mathbf{l}_m(1) = \mathbf{l}_m(T), \quad m \in \mathcal{M}\backslash\{0\}$

平均功率约束：

$\frac{1}{T} \sum_{t=1}^{T} p_m(t) \leq \bar{p}, \quad m \in \mathcal{M}\backslash\{0\}$

峰值功率约束：

$p_m(t) \leq p^{\max}_m, \quad p_n(t) \leq p^{\max}_n, \quad \forall m,n,t$

问题复杂度分析
该优化问题包含二进制变量 $b^{(i)}_{mn}(t)$、连续变量 $\mathbf{L}$ 和 $\mathbf{p}$，目标函数非凸且约束耦合。若采用穷举搜索，设空间离散化为 $L$ 个位置单元、功率离散化为 $P$ 个等级，则计算复杂度为：


$\mathcal{O}\left( (N \cdot L \cdot P)^{2M} \right)$