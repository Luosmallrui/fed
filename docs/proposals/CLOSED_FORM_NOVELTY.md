# 闭式解驱动的 novelty 与 motivation 探索

> 日期：2026-10-06
> 状态：理论推导 + 合成数值验证已完成；**尚未在 RoBERTa/LLM 上验证**。文中凡涉及真实效果的表述均为待检验预测。
> 配套文件：Introduction 草稿 [`intro_draft.tex`](intro_draft.tex)；验证脚本在 [`scripts/proposals/`](../../scripts/proposals/)。

## 0. 结论

推荐把两个创新点都升级为“先定义目标 → 推导闭式解”的形式，并且两者来自同一个目标：**在精确字节预算下，压缩同步后的模型与 dense FedAvg 的偏差最小**。

| | 现在的版本 | 升级后的版本 | 闭式解 |
|---|---|---|---|
| 创新点一（时间维度） | 双向 Top-k + EF | **时延一致误差反馈（DCEF）** | 残差传播算子 $\Lambda^\star = I-\Gamma$；客户端由“从虚拟模型 $x+e_i$ 开始本地训练”精确实现，服务器用 $v=g+(1-\hat\gamma)e_s$，$\hat\gamma$ 为闭式最小二乘估计 |
| 创新点二（空间维度） | 预训练谱 A（结果为负） | **通信感知固定基（CAB）** | $A^\star=\mathrm{TopEig}_r\big(\bar G^\top\bar G-\kappa\,\Sigma_{\mathrm{het}}\big)$，$\kappa$ 由字节预算、客户端数和本地步长闭式给出 |

两者都直接来自你现有数据中的两个“负面/不完整”结果，这就是 motivation：

1. EF 在 36 对里 32 对正向，但 Full-LoRA EF 18 个运行里只有 7 个严格稳定，“达到 C95 不等于保持”。推导表明，标准 EF 在“本地多步训练 + 稀疏传输延迟”下会重复累加同一段位移，从而产生过冲；延迟 $T\approx d/K$ 越长（预算越小）越严重。
2. 谱 A 没有稳定优于随机 A。推导表明，谱 A 是 $\min\lVert W_0(I-P)\rVert$ 的闭式解，优化的是预训练权重能量。这个目标与压缩聚合误差无关，所以其效果必然随任务变化。

**新颖性需要如实说明。** DCEF 的客户端一半与 CSER、step-ahead EF（SAEF / SA-PEF）机制几乎相同，必须定位为“给出唯一最优性推导并推广到双向”，而不是“提出新机制”。CAB 在 $\kappa=0$ 时退化为联邦版 gradient-PCA 初始化（LoRA-GA、SLoRA、Fed-SB 一类），真正的新内容是由预算和异质性推出的 $-\kappa\Sigma_{\mathrm{het}}$ 项。详见第 4 节。

---

## 1. 探索过程：试过什么、为什么放弃

| 候选形式化 | 闭式解 | 放弃或保留的原因 |
|---|---|---|
| 规范自由度平衡：$\min_G \lVert BG\rVert^2+\lVert G^{-1}A\rVert^2$ | 矩阵几何均值 | 与 LoRA-RITE、ScaledGD、FedSVD、FedRot-LoRA、“Gauge-Aware Server Representations”(2605.06733) 高度重叠；放弃 |
| 张量间字节分配（rate–distortion 反向注水） | 注水解 | 全局 Top-k 已是无权 L2 下的最优分配；你的 P2 tensor-balanced 已是负结果；属经典压缩工程；放弃 |
| 跨轮预算调度 $K_t\propto\eta_t^{2/3}$ | Lagrange 解 | 依赖松散上界；学习率恒定时退化为常数；Accordion 等已有；放弃 |
| 函数空间（$\Delta W$ 空间）残差搬运，Full LoRA | 切空间伪逆（Sylvester 方程闭式解） | 推导成立，但实现复杂、效果大小未知；留作后备 |
| **通信感知固定基 CAB** | Ky Fan：$\mathrm{TopEig}_r(M)$ | 保留。但合成实验显示：只有在“大步长 + 低预算”时 $\kappa$ 项才有用，而该区间里标准 EF 本身会发散，由此引出下一行 |
| **时延一致 EF（DCEF）** | 唯一解 $\Lambda^\star=I-\Gamma$ | 保留，是解释“达到 ≠ 保持”最直接的机制，也正好打开 CAB 生效的区间 |

关键转折是 CAB 的合成实验。纯 CAB 在标准 EF 的稳定区内与 $\kappa=0$ 几乎打平（60 个随机实例 31 胜 28 负）。在激进区间里，标准 Top-k+EF 对所有基都发散。追查发散原因，就得到了 DCEF。

---

## 2. 统一目标与两个闭式解

### 2.1 记号与精确字节预算

- 单向 dense 消息 $D_{\mathrm{full}}$，预算比例 $\beta\in\{0.1,0.2\}$，分类头包 $D_h$。
- 每个稀疏坐标花费 6 B（uint32 索引 + FP16 值），包头 21 B。
- 坐标数 $K(\beta)=\big\lfloor(\lfloor\beta D_{\mathrm{full}}\rfloor-21-D_h)/6\big\rfloor$；适配器维度为 $d$。
- 平均传输延迟 $T(\beta)\approx d/K(\beta)$。

| 参数化 | $d$ | $K$（10% / 20%） | 平均延迟 $T=d/K$（10% / 20%） | 标准 EF 稳定所需 $\gamma<2/T$（10% / 20%） |
|---|---:|---:|---:|---:|
| Full LoRA（A+B） | 294,912 | 9,363 / 19,245 | **31.5 / 15.3 轮** | 0.063 / 0.131 |
| Fixed-A（仅 B） | 147,456 | 9,363 / 19,245 | 15.7 / 7.7 轮 | 0.127 / 0.261 |

（$K$ 按 RTE 的 $D_{\mathrm{full}}=592{,}926$ B 计算，其他任务因分类头不同略有差别。）

这张表本身就是 motivation：**训练只有 30–50 轮，而 10% 预算下一个坐标平均要等 31.5 轮才能被发送一次。**

### 2.2 闭式解一：时延一致残差算子

**局部线性化假设。** 客户端 $i$ 从起点 $s$ 做 $\tau$ 步本地训练，得到
$$s\mapsto s+\Gamma_i(c_i-s),\qquad \Gamma_i=I-(I-\eta H_i)^\tau .$$
这对二次局部目标加 GD 精确成立，一般情况下为一阶近似。dense FedAvg 每轮使全局状态移动 $\Gamma(c-x)$。

**优化目标（时延一致性）。** 考虑一般的线性残差规则：$u_t=\Delta_t+\Lambda e_t$，$q_t=\mathcal C_K(u_t)$，$e_{t+1}=u_t-q_t$。要求任一坐标块在被扣留 $T$ 轮后送达时，送达量恰好等于 dense 协议在这 $T$ 轮内会产生的位移：
$$\sum_{s=0}^{T-1}\Lambda^{s}\,\Gamma\,(c-x)=\big[I-(I-\Gamma)^T\big](c-x),\qquad \forall\,T\ge1,\ \forall\,c,x .$$
也可以等价写成 $\min_\Lambda\sum_T w_T\lVert\cdot\rVert^2$，其最小值为 0。

**定理 1（唯一闭式解）。** 若 $\Gamma$ 可逆，满足上式的 $\Lambda$ 唯一，为
$$\boxed{\Lambda^\star=I-\Gamma=(I-\eta H)^\tau}$$
证明：取 $T=2$ 得 $(I+\Lambda)\Gamma=2\Gamma-\Gamma^2$，故 $\Lambda=I-\Gamma$。代入后，几何级数对任意 $T$ 都成立。

**推论 1（标准 EF 的失稳阈值）。** 标准 EF 即 $\Lambda=I$，送达量为 $T\Gamma(c-x)$，误差乘子为 $1-T\gamma$。因此
$$\text{标准 EF 稳定}\iff T\gamma<2\iff \gamma<\frac{2K(\beta)}{d}.$$
预算越小、本地训练每轮推进越多、$d$ 越大（Full LoRA 是 fixed-A 的 2 倍），就越容易过冲。两个端点：$\Gamma\to0$（小步、梯度型）时 $\Lambda^\star\to I$，即标准 EF；$\Gamma\to I$（本地训练收敛、位移型）时 $\Lambda^\star\to0$，即不保留残差。所以 **EF 与 no-EF 是同一族解的两个端点**。

**推论 2（客户端无需知道 $\Gamma$）。** 客户端从虚拟模型 $x+e_i$ 开始本地训练，并上传“终点 $-x$”，则
$$u=e+\Gamma(c-x-e)=\Gamma(c-x)+(I-\Gamma)e ,$$
自动实现 $\Lambda^\star$，对矩阵 $\Gamma$ 也成立。这一实现与 CSER、step-ahead EF 相同；我们给出的是它的最优性来源。

**推论 3（服务器端闭式折扣）。** 下行残差 $e_s$ 的问题在于：客户端在 $x$（不含 $e_s$）处重新计算的更新会再次请求 $\bar\Gamma e_s$。去重后
$$v=g+(I-\bar\Gamma)e_s\ \approx\ g+(1-\hat\gamma)e_s,\qquad \hat\gamma=\sum_ip_i\hat\gamma_i,\qquad \hat\gamma_i=-\frac{\langle d_{t+1}-d_t,\ s_{t+1}-s_t\rangle}{\lVert s_{t+1}-s_t\rVert^2},$$
其中 $d_t$ 为本地位移、$s_t$ 为本地起点。$\hat\gamma_i$ 是割线最小二乘的闭式解，每条消息多传一个 FP32 标量（+4 B，计入 exact-wire）。

### 2.3 闭式解二：通信感知固定基

**设定。** 固定行正交 $A\in\mathbb R^{r\times n}$，$P=A^\top A$，只训练并传输 $B$。客户端最优位移为 $\bar\Delta+D_i$，满足 $\sum_ip_iD_i=0$。分析压缩器取无偏 $\omega$-压缩器，各客户端独立，$\omega=d/K(\beta)-1$；有效客户端数 $N_{\mathrm{eff}}=1/\sum_ip_i^2$。

**优化目标（稳态超额误差）。** 衡量的是固定最后一轮的偏差，与你的主指标一致：
$$J(P)=\mathbb E\lVert W_\infty-W^\star\rVert_F^2=\underbrace{\lVert\bar\Delta(I-P)\rVert_F^2}_{\text{子空间外，无法表示}}+\kappa\,\underbrace{\mathrm{tr}(P\,\Sigma_{\mathrm{het}})}_{\text{异质性造成的压缩噪声底}},$$
$$\kappa=\frac{\gamma\,\omega/N_{\mathrm{eff}}}{2-\gamma(1+\omega/N_{\mathrm{eff}})},\qquad \Sigma_{\mathrm{het}}=N_{\mathrm{eff}}\sum_ip_i^2D_i^\top D_i .$$
直觉：dense 同步时，异质位移在平均中精确抵消；稀疏同步时，各客户端发送的坐标不同步，抵消不完全，在最优点附近留下噪声底。这正是“达到后又掉下来”的第二个来源。

**定理 2（Ky Fan 闭式解）。**
$$\boxed{A^\star=\mathrm{TopEig}_r\big(M\big),\qquad M=\bar\Delta^\top\bar\Delta-\kappa\,\Sigma_{\mathrm{het}}},\qquad r^\star=\min\{r,\ \#\{j:\lambda_j(M)>0\}\}.$$
实际估计时，用初始梯度代理：$\bar\Delta\propto-\bar G$，$D_i\propto-(G_i-\bar G)$。两者比例常数相同，所以特征向量与尺度无关：
$$A^\star=\mathrm{TopEig}_r\Big(\bar G^\top\bar G-\kappa\,N_{\mathrm{eff}}\textstyle\sum_ip_i^2(G_i-\bar G)^\top(G_i-\bar G)\Big).$$

**推论。**
- (a) $\beta\to1$（不压缩）时 $\kappa=0$，$A^\star$ 为共识梯度的前 $r$ 个右奇异向量，即联邦版 LoRA-GA / SLoRA 类初始化。
- (b) 谱 A 是 $\min_P\lVert W_0(I-P)\rVert$ 的解，与 $\bar\Delta$、$\Sigma_{\mathrm{het}}$ 无关。它只有在 $W_0$ 的主行空间恰好与 $M$ 的主特征空间对齐时才最优，这解释了 RTE 为负、MNLI 20% 为正的任务依赖。
- (c) $\kappa$ 随预算减小、本地步长增大、客户端数减少而增大。这些都是可证伪的预测。

**估计开销。** 每个客户端一次性上传各层梯度的秩 $k$ 截断 SVD，以平衡因子 $U\sqrt S$ 与 $\sqrt S V^\top$ 的形式用 FP16 传输（$k=r=8$ 时约等于一条 dense 适配器消息）；服务器据此重建低秩梯度、计算 $M$，并一次性下发 $A^\star$（每张量 $r\times n$，FP16）。全部计入 exact-wire。

最初设计用共享种子的高斯左 sketch $\Psi G_i$，在 768×768 的合成梯度上做了同字节对比（每层每客户端 24 KB）：截断 SVD 得到的目标值是精确最优的 1.004–1.07 倍，sketch 是 1.10–1.25 倍。原因是 sketch 以 $\sqrt{2/q}\approx0.35$ 的相对噪声扭曲各方向的权重，而减去 $\kappa\Sigma_{\mathrm{het}}$ 会放大这种扭曲。因此实现采用截断 SVD。实现细节见 [CAB_IMPLEMENTATION.md](CAB_IMPLEMENTATION.md)。

### 2.4 两者的关系

- DCEF 消除的是**时间维度**的误差（延迟导致的重复累加与过冲），使大本地步长、低预算的区间变得可用。
- CAB 消除的是**空间维度**的误差（不可表示的共识方向 + 异质噪声底）。
- 合成实验显示：只有在 DCEF 打开的激进区间里，CAB 的 $\kappa$ 项才有显著收益（第 3 节）。两者互补，不是简单拼接。

---

## 3. 数值验证（合成，仅验证推导与机制，不代表真实任务效果）

所有结果都可以用 `python3 scripts/proposals/<脚本>` 复现，纯 numpy，CPU 上 1–2 分钟。

| 检查 | 结果 | 脚本 |
|---|---|---|
| 单轮期望恒等式（无偏 rand-k） | 预测 954.507，Monte Carlo 954.464 ± 1.434 | `cab_synthetic_check.py` [1] |
| Ky Fan 最优性 | 闭式 620.155 = 20 次 Stiefel 梯度上升最优 620.155；5000 个随机子空间最优仅 259.7 | 同上 [2] |
| 稳态噪声底公式 | K = 13 / 26 / 51 时，理论 5.4340 / 5.3076 / 5.2662，仿真 5.4338 / 5.3075 / 5.2661 | 同上 [3] |
| 标准 EF 失稳阈值（理论值 2） | 最大稳定 $\gamma d/K=2.60$，最小失稳 $2.85$；无异质时 24 个设置中 14 个发散 | `dcef_synthetic_check.py` |
| DCEF 稳定性 | 48 个设置中 0 个发散（oracle 与客户端估计 $\hat\gamma$ 一致） | 同上 |
| 异质场景的尾部误差 / 保持率 | 标准 EF：中位尾误差 0.179、中位保持率 0.00；DCEF：0.031、1.00 | 同上 |
| 类 Adam（sign 下降）本地训练 | $\rho=3\%$（延迟 33 轮，对应你的 Full-LoRA 10%）：标准 EF 相对误差 2.14，不收敛；DCEF 与 dense 一致（3e-5） | `dcef_signdescent_check.py` |
| CAB vs $\kappa=0$，标准 EF 稳定区 | 31 胜 / 1 平 / 28 负，中位 +0.9%：**无差别**，与 $\kappa\propto\gamma$ 的预测一致 | `cab_synthetic_check.py` [6] |
| CAB vs $\kappa=0$，DCEF，$\rho$ = 5% / 10% | 57/60、54/60 胜，中位 +50% / +34%；**最差 −37% / −62%** | 同上 [6] |
| CAB vs 随机 / 谱基 | 三种设置中位 +43–45%，最差仍为正（+12% 以上） | 同上 [6] |
| 理论 $J$ 与仿真排序 | Spearman 0.80–0.83 | 同上 [6] |
| 协同（设计场景，DCEF） | $\rho=5\%$：$\kappa=0$ 为 3.48，理论 $\kappa$ 为 2.815（−19%）；$\rho=20\%$ 时 $\kappa\approx0$ 最好，与理论一致 | `dcef_basis_synergy_check.py` |

需要如实交代的局限：
1. 合成模型是二次目标、共享曲率、全参与，异质性按人为设定。
2. CAB 的 $\kappa$ 来自无偏压缩器模型，用于 Top-k + DCEF 时只是近似，最差实例仍会变差，需要做 $\kappa$ 倍率的鲁棒性实验。
3. 一切真实效果都还没有验证。

---

## 4. Novelty 定位（逐项扣除）

| 近邻 | 已做的内容 | 我们剩下的差异 | 风险 |
|---|---|---|---|
| [CSER](https://arxiv.org/abs/2007.13221)（NeurIPS 2020） | 本地模型吸收压缩误差，定期 error reset | 证明“从虚拟模型开始”是时延一致的**唯一**解；用于 FL 本地多步 + 双向；无需 reset | **高**：客户端机制几乎相同，必须正面引用 |
| SAEF（AAAI 2021）/ [SA-PEF](https://arxiv.org/abs/2601.20738)（TMLR 2026） | step-ahead EF；$\alpha\in[0,1]$ 可调，有理论建议值 | 在位移型更新下 $\alpha=1$ 精确最优；下行折扣；失稳阈值与字节预算挂钩 | **高** |
| [FedSpar](https://arxiv.org/abs/2307.10815) | 部分参与下用启发式 $\kappa$ 折扣陈旧残差 | 折扣 $1-\hat\gamma$ 由目标推导，并有闭式估计 | 中 |
| [Fed-EF](https://arxiv.org/abs/2211.14292)（ICML 2023） | 部分参与造成的 stale error 分析 | 稀疏延迟 $d/K$ 造成的过冲与失稳阈值 | 中 |
| [EControl](https://arxiv.org/abs/2311.05645)（ICLR 2024） | 控制 EF 信号强度 | 机制不同（梯度估计器），不针对 FL 本地多步 | 低–中 |
| LoRA-GA / [SLoRA](https://arxiv.org/abs/2308.06522) / [Fed-SB](https://arxiv.org/abs/2502.15436) | 梯度或预训练得到的子空间初始化 | $\kappa=0$ 即退化为它们；新内容是预算与异质性推出的 $-\kappa\Sigma_{\mathrm{het}}$ | 中高 |
| [FedLore](https://arxiv.org/abs/2610.01620)（2026-10-01） | 每轮共享并刷新低秩基，消除投影偏差 | 选基准则不含预算与异质惩罚，无 EF/字节 | 中 |
| [FedSAIL](https://arxiv.org/abs/2609.32485) / Dysco / FedGSA | 聚合 Gram 或投影的前 $r$ 个特征向量 | 准则是能量或一致度，而非由压缩方差推导 | 中 |
| [SpecTraL](https://arxiv.org/abs/2607.21074) | 用 spiked covariance 分离共识与非 IID 噪声，用于选秩 | 我们选的是方向，且 $\kappa$ 由预算给出 | 中 |
| FeDeRA / PiSSA / LoRA-XS / 你的 Spectral-A | 预训练权重谱 | 推论 (b)：错误目标的闭式解 | — |

**建议的贡献表述。** 不说“我们提出从虚拟模型开始的 EF”，而是：

> 我们把精确字节预算下的双向稀疏同步写成“与 dense FedAvg 轨迹一致”的优化问题，证明其残差算子有唯一闭式解。该解在客户端恰好给出 step-ahead 形式（解释了 CSER/SAEF 为何有效，并说明系数应取 1），在服务器端给出此前没有的闭式折扣，并推出标准 EF 的失稳阈值 $\gamma<2K(\beta)/d$。

---

## 5. 在你现有数据上的 motivation 证据与待检验预测

**已有、可以直接写进 motivation 的事实**（来自 `docs/results`）：

- EF 有效但不保持：Full-LoRA EF 18 个运行中 14 个达到 C95，13 个终轮保持，只有 7 个严格稳定。
- 预算越小越不稳：SST-2 严格稳定 10% 为 1/3，20% 为 2/3。
- 谱基与随机基：dense 下 Spectral − Random 为 SST-2 −0.08、RTE −4.21、MNLI +2.97 个点；稀疏 + EF 下仅 MNLI 20% 三个 seed 同向为正。

**待检验预测**（任何一条不成立都应如实修改叙事）：

| 编号 | 预测 | 检验方式 |
|---|---|---|
| P1 | 标准 EF 的命中后违约率、摆幅、回撤：10% > 20%，Full LoRA > fixed-A（延迟比为 2:1） | Phase 0，零 GPU |
| P2 | 出现命中后掉落的配置中，$\gamma_{\mathrm{eff}}\cdot d/K\gtrsim2$；或在 Adam 下残差/更新比随延迟增大 | Phase 0 + Phase 1 |
| P3 | DCEF 消除 10% 与 20% 之间的稳定性差距，严格稳定数上升，且不增加字节 | Phase 2 |
| P4 | 增加本地步数会使标准 EF 更不稳，但 DCEF 不会 | Phase 2 |
| P5 | CAB 相对 $\kappa=0$ 的收益随预算减小、Dirichlet $\alpha$ 减小、客户端数减少而增大，dense 时消失 | Phase 3 |
| P6 | Spectral − Random 的符号可由 $\mathrm{tr}(P_{\mathrm{spec}}M)-\tfrac{r}{n}\mathrm{tr}(M)$ 预测 | Phase 3（只需算梯度，不需训练） |

---

## 6. 验证计划与止损条件

**Phase 0（零 GPU，今天就能做）。** 本地副本没有逐轮 JSON，需要在服务器上运行：
```bash
python3 scripts/proposals/diagnose_ef_overshoot.py docs/results/fedavg_lora_wire docs/results/roberta_screen docs/results/fixed_a_ablation docs/results/full_lora_noef --out docs/proposals/ef_overshoot.csv
```
脚本输出每个运行的命中后违约率、摆幅、最大回撤、best−final、残差/更新比及其斜率，并按（方法，预算）汇总，用于检验 P1。

**Phase 1（约 1 GPU·小时）：测量 $\gamma_{\mathrm{eff}}$。**
- 在 dense FedAvg-LoRA 的第 5、15、25 轮，对若干客户端分别从 $x$ 和 $x+\delta$（$\lVert\delta\rVert$ 为单轮更新的 10–50%）各跑一次本地训练，固定相同的 minibatch 种子。
- 计算 $\gamma_{\mathrm{eff}}=-\langle\Delta(x+\delta)-\Delta(x),\delta\rangle/\lVert\delta\rVert^2$，再与上表的 $2/T$ 比较。
- 注意：AdamW 每轮重置时，本地更新接近 sign 型，$\gamma$ 的意义变弱。此时直接看 Phase 2 的 EF vs DCEF 更有说服力，合成实验已说明 sign 型更新同样会过冲。

**Phase 2（约 1–2 天 GPU）：实现 DCEF。** 已实现（2026-10-06），方法名、配置、报告脚本和运行方案见 [DCEF_IMPLEMENTATION.md](DCEF_IMPLEMENTATION.md)。以下为最初的补丁草图：
```python
# 客户端：从虚拟模型开始
start = global_adapter + client_residuals[i]
adapter_layout.assign(model, start); ... _train_local(...)
corrected = adapter_layout.flatten(model) - global_adapter   # 已隐含 (I-Γ)e
compressed = adapter_compressor.compress(corrected)
client_residuals[i] = corrected - compressed.reconstructed
# 同时更新割线统计并上传 gamma_i（+4 B）
# 服务器：
corrected = aggregate_adapter + (1.0 - gamma_hat) * server_residual
```
- 使用新的开发 seed（401–403 已经看过）。
- 矩阵：SST-2/RTE/MNLI × {5%, 10%, 20%} × {EF, DCEF} × 3 seeds，外加本地步数 {4, 8, 16} 的小扫描，用于检验 P4。
- 注意分类头目前是 dense 同步、没有 EF，DCEF 只作用于适配器，这一点要写清楚。

**Phase 3：CAB。** 已实现（2026-10-06），见 [CAB_IMPLEMENTATION.md](CAB_IMPLEMENTATION.md)。
- 先做 P6：只需在初始点算各客户端梯度，不需训练。
- 再比较 Random / Spectral / $\kappa=0$ / CAB（理论 $\kappa$ 及 0.5×、2×），均在 DCEF 下，预算 5/10/20%，Dirichlet $\alpha\in\{0.1,0.3,1.0\}$。

**Phase 4：LLM 规模**（投稿必须）。用 Qwen2.5-1.5B 或 7B 做指令或生成任务，配置为 DCEF + 最优基与强基线的对比。

**止损条件。**
- P1 不成立（10% 与 20% 没有过冲差异），且 Phase 2 中 DCEF ≤ EF：放弃 DCEF 叙事，回到实证论文路线。
- CAB 在所有区间都与 $\kappa=0$ 无差别：$-\kappa\Sigma$ 项不作为 novelty，只保留“谱 A 是错误目标”的解释。

---

## 7. 与毕业论文和投稿的对应

- **毕业论文**：两个创新点保持原编号，各自升级为“目标 → 闭式解 → 实证检验”。原有的谱 A 负结果和 EF 稳定性问题，正好作为各自的 motivation。
- **ICML（约 1 月底）**：以定理 1、推论 1–3 加 LLM 实验为主体；CAB 作为第二部分，或留给另一篇。
- **ACL（1 月 4 日 ARR）**：需要 LLM 任务（指令微调、生成），理论放精简版。
- **WWW（10 月 25 日）**：时间只够做 Phase 0–2 的一部分，不建议把这条新方法线赶进 WWW。
