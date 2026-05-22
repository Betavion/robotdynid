# 动力学参数辨识优化理论说明

本文档记录 `robotdynid` 当前离线辨识流程中的优化目标、参数分组和数值改进项。符号统一采用如下规范：

$$
\boldsymbol{\tau}
=
\boldsymbol{H}(\boldsymbol{q}, \dot{\boldsymbol{q}}, \ddot{\boldsymbol{q}}, \boldsymbol{q}_{si})
\boldsymbol{\xi}
+
\boldsymbol{\varepsilon}
$$

堆叠后的回归形式写为：

$$
\boldsymbol{\Gamma}
=
\boldsymbol{W}(\boldsymbol{q}, \dot{\boldsymbol{q}}, \ddot{\boldsymbol{q}}, \boldsymbol{q}_{si})
\boldsymbol{\xi}
+
\boldsymbol{\varepsilon}
$$

其中纯线性可辨识部分为：

$$
\boldsymbol{\tau}^{\mathrm{linear}}
=
\boldsymbol{H}(\boldsymbol{q}, \dot{\boldsymbol{q}}, \ddot{\boldsymbol{q}}, \boldsymbol{q}_{si})
\boldsymbol{\xi}
$$

线性参数向量为：

$$
\boldsymbol{\xi}
=
\begin{bmatrix}
\boldsymbol{\xi}_{\mathrm{BIP}}^\mathsf{T} &
\boldsymbol{\xi}_{\mathrm{JD}}^\mathsf{T}
\end{bmatrix}^\mathsf{T}
$$

## 1. 模型结构

对第 $k$ 个采样点，关节状态为：

$$
\boldsymbol{q}_k,\ \dot{\boldsymbol{q}}_k,\ \ddot{\boldsymbol{q}}_k
$$

测得关节力矩为：

$$
\boldsymbol{\tau}_k \in \mathbb{R}^{n}
$$

单点模型写为：

$$
\boldsymbol{\tau}_k
=
\boldsymbol{H}_k(\boldsymbol{q}_{si})
\boldsymbol{\xi}
+
\boldsymbol{\varepsilon}_k
$$

其中：

- $\boldsymbol{H}_k(\boldsymbol{q}_{si})$ 是第 $k$ 个采样点的动力学回归矩阵；
- $\boldsymbol{\xi}$ 是所有线性可辨识参数；
- $\boldsymbol{q}_{si}$ 是 Stribeck 非线性速度尺度参数；
- $\boldsymbol{\varepsilon}_k$ 是未建模项、外部扰动和测量误差等合并后的残差。

残差定义为：

$$
\boldsymbol{\varepsilon}_k
=
\boldsymbol{\tau}_k
-
\boldsymbol{H}_k(\boldsymbol{q}_{si})
\boldsymbol{\xi}
$$

线性参数分成 BIP 和 JD 两组：

$$
\boldsymbol{\xi}
=
\begin{bmatrix}
\boldsymbol{\xi}_{\mathrm{BIP}} \\
\boldsymbol{\xi}_{\mathrm{JD}}
\end{bmatrix}
$$

其中 $\boldsymbol{\xi}_{\mathrm{BIP}}$ 是 base inertial parameters，$\boldsymbol{\xi}_{\mathrm{JD}}$ 是 joint dynamics 参数。当前默认启用：

$$
\boldsymbol{\xi}_{\mathrm{JD}}
=
\begin{bmatrix}
\boldsymbol{f}_v^\mathsf{T} &
\boldsymbol{f}_c^\mathsf{T} &
\boldsymbol{f}_d^\mathsf{T}
\end{bmatrix}^\mathsf{T}
$$

分别对应粘性摩擦、库仑摩擦和 Stribeck 幅值。Stribeck 速度尺度 $\boldsymbol{q}_{si}$ 不属于线性参数，而是进入回归矩阵：

$$
H_{fd,i}(\dot q_i;q_{si,i})
=
\operatorname{sign}(\dot q_i)
\exp\left(-\left|\frac{\dot q_i}{q_{si,i}}\right|\right)
$$

## 2. BIP 选择

先使用惯性参数标准回归矩阵：

$$
\boldsymbol{Y}_s\boldsymbol{\xi}_s
\approx
\boldsymbol{\Gamma}
$$

通过带列主元 QR 分解选择独立列：

$$
\boldsymbol{Y}_s\boldsymbol{P}
=
\boldsymbol{Q}\boldsymbol{R}
$$

独立列数由 QR 对角线和 SVD 校验确定。保留的独立列形成 BIP 回归结构：

$$
\boldsymbol{Y}_b\boldsymbol{\xi}_{\mathrm{BIP}}
$$

当前实现中，BIP 结构在正式辨识前确定；后续 Stribeck 交替优化不会反复重选 BIP 列。

## 3. 堆叠回归问题

所有采样点堆叠后得到：

$$
\boldsymbol{\Gamma}
=
\begin{bmatrix}
\boldsymbol{\tau}_1 \\
\boldsymbol{\tau}_2 \\
\vdots \\
\boldsymbol{\tau}_N
\end{bmatrix},
\quad
\boldsymbol{W}(\boldsymbol{q},\dot{\boldsymbol{q}},\ddot{\boldsymbol{q}},\boldsymbol{q}_{si})
=
\begin{bmatrix}
\boldsymbol{H}_1(\boldsymbol{q}_{si}) \\
\boldsymbol{H}_2(\boldsymbol{q}_{si}) \\
\vdots \\
\boldsymbol{H}_N(\boldsymbol{q}_{si})
\end{bmatrix}
$$

因此：

$$
\boldsymbol{\Gamma}
=
\boldsymbol{W}(\boldsymbol{q},\dot{\boldsymbol{q}},\ddot{\boldsymbol{q}},\boldsymbol{q}_{si})
\boldsymbol{\xi}
+
\boldsymbol{\varepsilon}
$$

为避免符号冲突，本文档后续用 $\boldsymbol{\Lambda}$ 表示测量白化/加权矩阵；$\boldsymbol{W}$ 始终表示堆叠后的动力学回归矩阵。

## 4. 测量协方差权重

若认为每个关节力矩噪声独立，且标准差为：

$$
\boldsymbol{s}
=
\begin{bmatrix}
s_1 & \cdots & s_n
\end{bmatrix}^\mathsf{T}
$$

则每个采样点的测量协方差为：

$$
\boldsymbol{R}
=
\operatorname{diag}(s_1^2,\dots,s_n^2)
$$

堆叠后的白化矩阵记为：

$$
\boldsymbol{\Lambda}
=
\operatorname{diag}(\boldsymbol{R}^{-1/2},\boldsymbol{R}^{-1/2},\dots,\boldsymbol{R}^{-1/2})
$$

给定 $\boldsymbol{q}_{si}$ 时，线性参数的数据项为：

$$
\min_{\boldsymbol{\xi}}
\left\|
\boldsymbol{\Lambda}
\left(
\boldsymbol{W}(\boldsymbol{q},\dot{\boldsymbol{q}},\ddot{\boldsymbol{q}},\boldsymbol{q}_{si})
\boldsymbol{\xi}
-
\boldsymbol{\Gamma}
\right)
\right\|_2^2
$$

实现中支持两类权重叠加：

- `torque_weighting: torque_std`：根据数据中各关节实测力矩标准差自动归一化；
- `measurement_torque_std`：用户指定每关节力矩噪声标准差，对应显式对角协方差。

如果两者同时启用，最终行权重为二者乘积。

## 5. 线性参数先验正则化

为了降低病态回归矩阵、噪声和异常采样对参数的影响，引入线性参数先验：

$$
\boldsymbol{\xi}
\sim
\mathcal{N}(\boldsymbol{\xi}_0,\boldsymbol{\Sigma}_{\xi})
$$

则 MAP 估计目标为：

$$
\min_{\boldsymbol{\xi}}
\left\|
\boldsymbol{\Lambda}
\left(
\boldsymbol{W}\boldsymbol{\xi}
-
\boldsymbol{\Gamma}
\right)
\right\|_2^2
+
\lambda
\left\|
\boldsymbol{\Sigma}_{\xi}^{-1/2}
(\boldsymbol{\xi}-\boldsymbol{\xi}_0)
\right\|_2^2
$$

其中：

- $\boldsymbol{\xi}_0$ 是线性参数先验；
- $\boldsymbol{\Sigma}_{\xi}$ 是线性参数先验协方差；
- $\lambda$ 是 `linear_regularization_strength`。

当只给定每个参数的先验标准差 $d_i$ 时：

$$
\boldsymbol{\Sigma}_{\xi}
=
\operatorname{diag}(d_1^2,\dots,d_p^2)
$$

当不提供协方差或标准差时，默认使用各向同性 Tikhonov 正则：

$$
\min_{\boldsymbol{\xi}}
\left\|
\boldsymbol{\Lambda}
(\boldsymbol{W}\boldsymbol{\xi}-\boldsymbol{\Gamma})
\right\|_2^2
+
\lambda
\left\|
\boldsymbol{\xi}-\boldsymbol{\xi}_0
\right\|_2^2
$$

工程实现等价于增广线性系统：

$$
\begin{bmatrix}
\boldsymbol{\Lambda}\boldsymbol{W} \\
\sqrt{\lambda}\boldsymbol{\Sigma}_{\xi}^{-1/2}
\end{bmatrix}
\boldsymbol{\xi}
\approx
\begin{bmatrix}
\boldsymbol{\Lambda}\boldsymbol{\Gamma} \\
\sqrt{\lambda}\boldsymbol{\Sigma}_{\xi}^{-1/2}\boldsymbol{\xi}_0
\end{bmatrix}
$$

当前先验来源：

- `zero`：所有线性参数先验为 0；
- `urdf`：BIP 部分来自 URDF 惯性参数投影，JD 部分为 0；
- 显式 `linear_regularization_prior`：用户直接指定完整线性参数先验。

## 6. 鲁棒损失与 IRLS

普通最小二乘对异常点敏感。当前实现支持对数据残差引入 M 估计鲁棒损失：

$$
\min_{\boldsymbol{\xi}}
\sum_i f^2
\rho\left(
\left(\frac{r_i}{f}\right)^2
\right)
+
\lambda
\left\|
\boldsymbol{\Sigma}_{\xi}^{-1/2}
(\boldsymbol{\xi}-\boldsymbol{\xi}_0)
\right\|_2^2
$$

其中：

$$
\boldsymbol{r}
=
\boldsymbol{\Lambda}
\left(
\boldsymbol{W}\boldsymbol{\xi}
-
\boldsymbol{\Gamma}
\right)
$$

$f$ 是 `robust_f_scale`。支持的损失函数与 SciPy `least_squares` 对齐：

- `linear`
- `soft_l1`
- `huber`
- `cauchy`
- `arctan`

线性参数求解使用 IRLS。第 $j$ 轮根据上一轮残差计算鲁棒权重：

$$
\omega_i^{(j)}
=
\sqrt{
\rho'\left(
\left(\frac{r_i^{(j-1)}}{f}\right)^2
\right)
}
$$

然后求解：

$$
\min_{\boldsymbol{\xi}}
\left\|
\boldsymbol{\Omega}^{(j)}
\boldsymbol{\Lambda}
(\boldsymbol{W}\boldsymbol{\xi}-\boldsymbol{\Gamma})
\right\|_2^2
+
\lambda
\left\|
\boldsymbol{\Sigma}_{\xi}^{-1/2}
(\boldsymbol{\xi}-\boldsymbol{\xi}_0)
\right\|_2^2
$$

其中：

$$
\boldsymbol{\Omega}^{(j)}
=
\operatorname{diag}(\omega_i^{(j)})
$$

`robust_max_iterations` 控制 IRLS 最大轮数。`linear` 损失退化为普通加权最小二乘。

## 7. Stribeck 交替优化

由于 Stribeck 速度尺度 $\boldsymbol{q}_{si}$ 非线性进入回归矩阵，当前采用变量投影式交替优化。

给定 $\boldsymbol{q}_{si}$，先求：

$$
\boldsymbol{\xi}^*(\boldsymbol{q}_{si})
=
\arg\min_{\boldsymbol{\xi}}
J(\boldsymbol{\xi},\boldsymbol{q}_{si})
$$

然后对 Stribeck 参数做非线性更新：

$$
\boldsymbol{q}_{si}^*
=
\arg\min_{\boldsymbol{q}_{si}}
\left\|
\boldsymbol{\Lambda}
\left(
\boldsymbol{W}(\boldsymbol{q},\dot{\boldsymbol{q}},\ddot{\boldsymbol{q}},\boldsymbol{q}_{si})
\boldsymbol{\xi}^*(\boldsymbol{q}_{si})
-
\boldsymbol{\Gamma}
\right)
\right\|_2
$$

若启用鲁棒损失，外层 `scipy.optimize.least_squares` 也使用相同的 robust loss 和 `f_scale`。这样可以同时降低异常点对 Stribeck 参数和线性参数的影响。

完整流程为：

1. 选择 BIP 结构；
2. 初始化 $\boldsymbol{q}_{si}$；
3. 固定 $\boldsymbol{q}_{si}$，求解带权重、正则和可选鲁棒损失的线性参数 $\boldsymbol{\xi}$；
4. 固定线性求解器形式，更新 $\boldsymbol{q}_{si}$；
5. 重复 3-4，直到 Stribeck 参数变化或目标变化低于阈值，或达到最大迭代次数。

## 8. 配置建议

保守默认配置：

```yaml
identification:
  torque_weighting: torque_std
  measurement_torque_std: []
  linear_regularization_strength: 0.0
  linear_regularization_prior_source: zero
  linear_regularization_prior_std: []
  robust_loss: linear
  robust_f_scale: 1.0
  robust_max_iterations: 5
```

真实采集数据噪声较明显时，可先尝试：

```yaml
identification:
  torque_weighting: torque_std
  measurement_torque_std: [0.3, 0.5, 0.5, 0.1, 0.1, 0.1]
  linear_regularization_strength: 0.001
  linear_regularization_prior_source: urdf
  robust_loss: huber
  robust_f_scale: 1.0
  robust_max_iterations: 5
```

调参原则：

- 如果参数数值明显发散，先增大 `linear_regularization_strength`；
- 如果个别尖峰样本导致预测图异常，优先尝试 `huber` 或 `soft_l1`；
- 如果某些关节力矩噪声明显更大，设置 `measurement_torque_std`；
- 如果 URDF 惯性参数只是粗略值，`linear_regularization_strength` 不宜过大，避免把结果锁死在错误先验附近；
- `robust_f_scale` 应和加权后残差量级一致，过小会过度抑制正常动态样本。

## 9. 当前边界

当前优化仍然不是完整物理一致性约束优化：

- 未强制质量为正；
- 未强制惯性矩阵正定；
- 未对摩擦参数设置符号约束；
- BIP 结构选择完成后不随 Stribeck 参数重选；
- 全协方差测量噪声暂未建模为非对角 $\boldsymbol{R}$，工程入口主要支持对角关节噪声。

这些能力可以作为后续严格物理辨识阶段继续扩展。
