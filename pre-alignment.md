### 符号定义

+ **主轴上的两个方向：** $\mathbf{a^1_i}$，$\mathbf{a^2_i}$，其中$\mathbf{a^1_i}=-\mathbf{a^2_i}$

+ **次轴上的两个方向：** $\mathbf{b^1_i}$，$\mathbf{b^2_i}$，其中$\mathbf{b^1_i}=-\mathbf{b^2_i}$

+ **断面的点集：** $X_i$

+ **断面上的控制点：** $C_{X_i}$

+ **近邻函数（表示两个点集之间的平均距离）：** $dist(\cdot, \cdot)$

其中$i=1,2,\cdots,n $ 表示$n$个断面中第$i$个断面

### 优化目标函数

$$min \qquad dist(C_{X_i}, \mathbf\pi(C_{X_j}))$$

$$
s.t. \qquad \left\{
        \begin{array}{lr}
        \mathbf{a^k_i}=\lambda_k\mathbf{a^k_j}, &  \\
        \mathbf{b^k_i}=\mu_k\mathbf\pi(\mathbf{b^k_j}), &  \\
        \mathbf{a^k_i} \times \mathbf\pi(\mathbf{b^k_i})=-\mathbf{a^k_j} \times \mathbf\pi(\mathbf{b^k_j}). &
        \end{array}\right.
$$

$$1 \leq i,j\leq n,i\neq j \qquad k=1,2$$


### 其他
+ $C_{X_i}$由每个断面点集$X_i$经过k-means聚类得到的结果，假设聚类的数目为$n_c$，则$\left| C_{X_i} \right|=n_c$

+ $\mathbf\pi(\cdot)$表示点集或者向量经过某一个矩阵的变换后，得到新的点集或者向量

### 示意图
![](schematic_plot.jpg)