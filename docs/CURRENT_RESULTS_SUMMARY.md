# 现有实验结果、预期目标与补充实验说明

> 更新日期：2026-09-14  
> 适用对象：当前 RoBERTa-base / GLUE 联邦 PEFT 项目  
> 证据边界：本文只总结已经落盘并可审计的结果；开发实验、smoke test 和正式结果严格分开。

## 1. 一句话结论

当前实验已经可靠证明：在精确统计上行、下行和协议开销的条件下，LoRA 更新可以被压缩到约 10%–20% 的序列化通信量，并在 SST-2 和 MNLI 的部分设置中保留接近 dense FedAvg-LoRA 的性能。新增 fixed-A 机制消融进一步表明，双向 error feedback 在相同 wire 下通常能修复 Top-k 损失；但是，原先预期的“Spectral-A 在同等 wire 下稳定优于 Random-A”并未成立，V2 候选和 V4 的 Hybrid4+late-decay 也均未通过预注册门槛。因此，目前最稳妥的论文结论是“通信压缩与 EF 通常有效，但收益强烈依赖任务、预算和 seed；谱基优势及跨 seed 稳定性尚未得到证明”，而不是“已经得到一个稳定优于基线的新方法”。

## 2. 哪些材料是当前有效证据

| 材料 | 状态 | 可否用于效果结论 |
|---|---|---|
| 原始六方法矩阵 | 81/81 VALID，正式结果 | 可以 |
| FedAvg-LoRA exact-wire 补充矩阵 | 27/27 VALID，正式补充结果 | 可以 |
| Stability-v2 Stage A | 36/36 VALID，开发集负结果 | 只能用于方法开发和失败分析 |
| Stability-v3 | 4/30 后终止，已标记 `PARTIAL_SUPERSEDED` | 不可以 |
| Stability-v4 smoke | 10/10 完成，但仅 2 轮、seed 311 | 只能证明代码链路可运行 |
| Stability-v4 Stage A | 45/45 VALID，开发集负结果 | 只能用于方法开发和失败分析 |
| Fixed-A basis/EF 消融 | 54/54 VALID，official-validation post-hoc 机制证据 | 可以用于描述性机制归因，不能当作新盲确认 |
| Task-suite-v2 误启动 | 1 项完成、1 项中断，`PARTIAL_ABORTED_UNAUTHORIZED` | 不可用于效果结论，已隔离且禁止恢复 |
| 根目录 `main.tex` | 旧 Qwen2.5-0.5B、Banking77、4 客户端版本 | 不可引用其中旧结果 |

当前结果入口见 [正式结果汇总](results/RESULTS.md)、[结果分析](results/ANALYSIS.md) 和 [结果索引](results/README.md)。旧稿与新实验的冲突已经在 [ACL 论文计划](ACL_PAPER_PLAN.md) 中明确说明：旧稿只能保留动机，方法和结果必须按 RoBERTa/GLUE 证据重写。

## 3. 数据集与实验设置

### 3.1 正式实验数据集

| 数据集 | 任务 | 本地记录规模 | 当前用途 | 主指标 | 结果中暴露的问题 |
|---|---|---:|---|---|---|
| SST-2 | 二分类情感分析 | train 67,349；validation 872 | 81-run 主矩阵、27-run 补充矩阵 | official validation accuracy | dense 结果较稳定，但低通信方法对部分划分 seed 很敏感 |
| RTE | 二分类文本蕴含 | train 2,490；validation 277 | 81-run 主矩阵、27-run 补充矩阵、Stability-v2 | official validation accuracy | 训练集小、方差高，是当前最明显的失败场景 |
| MNLI | 三分类自然语言推断 | train 392,702；validation matched 9,815、mismatched 9,832 | 81-run 主矩阵、27-run 补充矩阵 | matched 与 mismatched validation accuracy 的均值 | seed 403 的 dense reference 退化到接近随机水平，显著影响均值和 C95 |

正式配置采用 RoBERTa-base，输入最大长度 128；训练集按标签使用 Dirichlet `alpha=0.3` 划分到 10 个客户端，每轮 10 个客户端全部参与。每个客户端 shard 的 10% 留作开发诊断，正式主指标使用 official GLUE validation。正式 partition seeds 为 401/402/403。具体协议见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) 和 [roberta_screen.yaml](../configs/roberta_screen.yaml)。

这些规模来自冻结的数据契约和 preflight。论文最终表格仍应从冻结的数据 manifest 自动导出，而不要手工维护，以保证数据版本、行数和 hash 一致。

### 3.2 新一轮 V4 数据集

V4 将后续主任务定义为 SST-2、MNLI、MRPC 和 QQP；Stage A 实际运行
SST-2/MRPC/QQP，MNLI 原定仅在通过 Stage A 后进入 Stage B。MRPC 是句对释义
识别，QQP 是重复问题识别；两者的冻结主指标都是
`(accuracy + positive-class F1) / 2`。数据契约记录 MRPC 为 3,668/408 条
train/validation，QQP 为 363,846/40,430 条 train/validation。

V4 Stage A 已完成 45/45 个 train-holdout development 运行。它可用于判定候选门禁，
但不是 official-validation 确认结果。由于候选仅 5/9 通过，MNLI Stage B
和确认集均未运行。

## 4. 方法分别做了什么

所有正式方法共享冻结的 RoBERTa-base 编码器、rank 8、LoRA alpha 16、query/value 投影、线性分类头和 AdamW。差别主要在于训练哪些 LoRA 参数，以及如何传输这些参数。

| 方法 | 核心做法 | 通信方式 | 在实验中的角色 |
|---|---|---|---|
| FedAvg-LoRA | A、B 两个 LoRA 因子都训练 | A+B 与分类头 dense 同步 | 性能和通信参考 |
| FFA-LoRA | 固定标准随机初始化 A，只训练 B | B 与分类头自然 dense 同步 | 约一半 LoRA 通信量的固定因子基线 |
| LoRA-XS | 固定预训练权重的 SVD 左/右子空间，只训练 `r×r` core | 极小的自然 dense 同步 | 极端参数/通信效率基线 |
| FLASC-style | 本地训练完整 LoRA，上下行分别做 Top-k，不保留 residual | 10%/20% exact-wire | 已有稀疏方法的受控基线；主指标是 masked-server 状态 |
| Random-A+EF | 固定行正交随机 A，只训练 B | 上下行 Top-k + 双向 error feedback | 与谱基严格匹配的随机子空间对照 |
| Spectral-A+EF | 固定预训练权重的前 r 个右奇异向量作为 A，只训练 B | 与 Random-A+EF 完全相同 | 原始拟议方法，用于检验谱子空间是否更有效 |
| FedAvg-LoRA+Top-k+EF | 与 dense FedAvg 一样训练完整 A+B，再联合压缩 A+B | 上下行 Top-k + 双向 EF | 隔离“完整 LoRA 在严格通信 cap 下的压缩代价” |
| Random/Spectral-A dense | 固定对应 A、只训练 B，不压缩 | B 与分类头自然 dense 同步 | Post-hoc 隔离固定子空间本身的效果 |
| Random/Spectral-A+Top-k | 固定对应 A、只训练 B，不保留 residual | 与 EF 版本完全相同的 10%/20% exact-wire | Post-hoc 隔离双向 EF 的增量 |

分类头始终 dense 同步且计入每个消息上限。稀疏值使用 FP16，索引使用 uint32；10%/20% 指的是完整序列化消息字节上限，不是参数坐标密度。完整 A+B 时，10%/20% wire 实际只允许约 3.10%–3.17% / 6.46%–6.53% 的 LoRA 坐标，因为索引、包头和 dense head 也占字节。

Stability-v2 另测试了三个后续候选：

- P1 `Spectral rotation + EF`：在不改变谱子空间的前提下旋转坐标，检验坐标对 Top-k/Adam 的影响。
- P2 `Tensor-balanced Top-k + EF`：按 LoRA-B tensor 分配 Top-k 配额，检验跨层支持饥饿。
- P3 `Hybrid4-A + EF`：保留 4 个谱方向，再加入 4 个正交随机方向，检验固定谱子空间容量不足。

## 5. 现有结果分别说明了什么

### 5.1 Dense 与自然通信基线

以下是三个正式 seed 的固定最终轮结果，数值为均值 ± 样本标准差。

| Task | FedAvg-LoRA | FFA-LoRA | LoRA-XS |
|---|---:|---:|---:|
| SST-2 | 0.9163 ± 0.0132 | 0.8987 ± 0.0384 | 0.7183 ± 0.1821 |
| RTE | 0.6823 ± 0.0253 | 0.6751 ± 0.0157 | 0.5560 ± 0.0201 |
| MNLI | 0.6593 ± 0.2954 | 0.7091 ± 0.1511 | 0.3334 ± 0.0181 |

这组结果说明：

1. FFA-LoRA 是当前相对可靠的折中基线。它只同步 B，通信量约为完整 LoRA 的 50.3%，在 SST-2 和 RTE 上接近 dense FedAvg。
2. “参数少”不等于“效果好”。LoRA-XS 的通信量只有 FedAvg 的约 1.0%–1.3%，但 SST-2/RTE 明显下降，在 MNLI 上接近随机水平。
3. MNLI 的三 seed 均值不能单独解读。FedAvg 的 seed 401/402 分别为 0.8274/0.8324，但 seed 403 只有 0.3182；FFA 在 seed 403 上反而为 0.5353。因此，FFA 的 MNLI 均值高于 FedAvg 不能直接写成稳定优越。

### 5.2 Spectral-A 是否优于 Random-A

正式最终准确率如下，每个单元均为 10% / 20% wire 下的三 seed 均值：

| Task | FLASC-style | Random-A+EF | Spectral-A+EF |
|---|---:|---:|---:|
| SST-2 | 0.7703 / 0.7894 | 0.8307 / 0.8375 | 0.7653 / 0.8444 |
| RTE | 0.4934 / 0.4910 | 0.6005 / 0.6185 | 0.5523 / 0.5656 |
| MNLI | 0.3835 / 0.4606 | 0.6592 / 0.6531 | 0.6386 / 0.7000 |

FLASC-style 的数值使用 masked-server 状态，只能单独描述，不能与后两者作同状态因果比较。真正回答谱基问题的是下表中完全匹配的 Spectral-A 与 Random-A 配对：

| Task | Wire | Spectral − Random | 配对方向 | 结论 |
|---|---:|---:|---:|---|
| SST-2 | 10% | −0.0654 ± 0.1007 | 1 正 / 2 负 | MIXED |
| SST-2 | 20% | +0.0069 ± 0.0212 | 2 正 / 1 负 | MIXED |
| RTE | 10% | −0.0481 ± 0.0292 | 0 正 / 3 负 | HURT |
| RTE | 20% | −0.0529 ± 0.0553 | 1 正 / 2 负 | MIXED |
| MNLI | 10% | −0.0206 ± 0.0358 | 1 正 / 2 负 | MIXED |
| MNLI | 20% | +0.0469 ± 0.0787 | 3 正 / 0 负 | SUPPORTED（仅此条件） |

这组实验直接回答了原始核心问题：谱基没有表现出跨任务、预算和 seed 的稳定优势。唯一一致正向的是 MNLI 20%；RTE 10% 则三个 seed 全部更差。因而可以说“谱基效果具有明显条件依赖”，不能说“谱基普遍优于随机基”。

原正式矩阵当时缺少 dense 与 no-EF 对照；这一缺口现已由下一节的 54-run
post-hoc 机制消融补齐。仍未补齐的是 uplink-only/downlink-only 分解和 Full-LoRA
no-EF，因而不能把 fixed-A 结果外推为所有 LoRA 参数化下的完整因果结论。

### 5.3 Fixed-A basis/EF 机制消融

新增 54 个运行使用冻结正式源码和 official validation，覆盖三个历史正式任务、
seeds401/402/403、Random/Spectral fixed-A dense，以及 10%/20% no-EF Top-k。
它们与原正式 36 个 EF 单元通过了零不匹配联合审计；每个 EF/no-EF 对的实际 wire
完全相同。由于 seeds 已经暴露，这些结果只能作为 post-hoc 描述性机制证据。

无压缩时的基底差异为：

| Task | Spectral dense − Random dense | Seed 方向 | 判断 |
|---|---:|---:|---|
| SST-2 | −0.000765 ± 0.013390 | 1 正 / 2 负 | TIE |
| RTE | −0.042118 ± 0.070405 | 1 正 / 2 负 | MIXED |
| MNLI | +0.029730 ± 0.043995 | 3 正 / 0 负 | SUPPORTED_POSITIVE，但受 seed403 放大 |

EF 相对同 basis、同预算 no-EF 的最终主指标增量为：

| Task | Wire | Random-A：EF − no-EF | Spectral-A：EF − no-EF |
|---|---:|---:|---:|
| SST-2 | 10% | +0.065749 ± 0.099984（3/3 正） | +0.118119 ± 0.171010（2 正/1 平） |
| SST-2 | 20% | +0.068807 ± 0.110277（3/3 正） | +0.091743 ± 0.114535（3/3 正） |
| RTE | 10% | +0.048135 ± 0.036520（3/3 正） | +0.062575 ± 0.057119（2 正/1 平） |
| RTE | 20% | +0.063779 ± 0.059794（3/3 正） | +0.061372 ± 0.046931（3/3 正） |
| MNLI | 10% | +0.052917 ± 0.035656（3/3 正） | +0.008603 ± 0.091856（2 正/1 负） |
| MNLI | 20% | −0.007856 ± 0.063336（2 正/1 负） | +0.051826 ± 0.029535（3/3 正） |

36 个 EF/no-EF 对合计 **32 正、2 平、2 负**。因此，最直接的机制结论是：
双向 EF 通常能在不增加实际 wire 的情况下修复 Top-k 损失，但不是必然有效。
no-EF residual 全部严格为 0，配对使用的 EF 对照最大相对守恒误差为
`7.211e-7 < 1e-6`，所以该差异
不是残差开关失效或额外通信造成的。

谱基在 no-EF 下仍没有普遍优势：SST-2 10%/20% 的 Spectral−Random 分别为
−0.117737/−0.016055，RTE 为 −0.062575/−0.050542（两个预算均 3/3 负），
MNLI 为 +0.023703/−0.012792。Basis×EF difference-in-differences 只有 SST-2
20%（+0.022936）和 MNLI 20%（+0.059682）三个 seed 同向为正；MNLI 后者又
明显受异常 seed403 放大。

两个重要失败点必须保留：SST-2 seed402 的四个 no-EF 稀疏单元都约为
0.509174；MNLI seed403 会显著放大 dense 与 20% EF 的谱基均值。实验也没有
拆分上行/下行 EF，故这里只能归因“客户端+服务器双向 EF 的联合效果”。

### 5.4 完整 LoRA 的 exact-wire 压缩

| Task | Budget | 最终准确率 | 相对 dense 的配对变化 | 达到 C95 | 平均 Speedup95 | 最终保持 | 严格稳定 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SST-2 | 10% | 0.9029 ± 0.0411 | −0.0134 ± 0.0286 | 3/3 | 4.996× | 2/3 | 1/3 |
| SST-2 | 20% | 0.9178 ± 0.0163 | +0.0015 ± 0.0035 | 3/3 | 3.148× | 3/3 | 2/3 |
| RTE | 10% | 0.6065 ± 0.0524 | −0.0758 ± 0.0761 | 1/3 | 5.552×，仅 n=1 | 1/3 | 0/3 |
| RTE | 20% | 0.6306 ± 0.0362 | −0.0517 ± 0.0481 | 1/3 | 3.471×，仅 n=1 | 1/3 | 0/3 |
| MNLI | 10% | 0.6614 ± 0.2669 | +0.0021 ± 0.0287 | 3/3 | 7.435× | 3/3 | 2/3 |
| MNLI | 20% | 0.6604 ± 0.2875 | +0.0010 ± 0.0091 | 3/3 | 4.105× | 3/3 | 2/3 |

这组结果是当前最清晰的正向证据：完整 LoRA 经 Top-k+EF 后，在 SST-2 20% 以及 MNLI 10%/20% 上，按三 seed 配对均值与 dense 基本相当，并用更少累计 wire 达到配对 dense 最终性能的 95%。但它也说明了三个边界：

1. RTE 上压缩方案不可靠，只有 seed 403 达到 C95。
2. “曾达到目标”不等于“最后仍保持目标”。18 个压缩单元中 14 个曾达到 C95，13 个最终仍达标，但仅 7 个满足完整的严格稳定条件。
3. 10% 与 20% 是效率—稳定性折中。20% 往往更稳定，但每轮消息更大，因此 Speedup95 反而可能更低。

MNLI 的三 seed Speedup95 还会被异常低的 seed-403 dense reference 放大。只看健康 seeds 401/402 时，10% 和 20% 的 Speedup95 分别为 6.185× 和 3.665×，对应最终准确率损失 0.0144 和 0.0036；正文必须同时给出完整三 seed 和该敏感性分析。

### 5.5 Stability-v2 为什么是负结果

Stability-v2 使用未见的开发 seeds 321/322/323，在 SST-2、RTE、10% wire 上测试 P1/P2/P3。所有 36 个运行均通过技术审计，但三个候选均未达到 6/6 strict-stable 门槛。

| 候选 | 达到 C95 | Strict stable | 最差 retention | 最差 TailDrop | 结论 |
|---|---:|---:|---:|---:|---|
| P1 Spectral rotation + EF | 4/6 | 4/6 | 0.7109 | 0.0000 | 淘汰 |
| P2 Tensor-balanced Top-k + EF | 3/6 | 3/6 | 0.6271 | 0.0000 | 淘汰 |
| P3 Hybrid4-A + EF | 6/6 | 3/6 | 0.8919 | 0.0803 | 淘汰 |

P3 最有信息量：它让 6/6 单元都曾达到 C95，但仍有三个单元因最终回落、命中后再次跌破或 TailDrop 超限而失败。这说明当前主要瓶颈已经不只是“能否快速到达较好精度”，而是“压缩训练能否在后期稳定维持该精度”。

所有 packet cap、EF 守恒、basis 正交性、冻结参数和 provenance 检查都通过，因此这些失败不能由已审计的丢包、超预算或基底漂移解释；下一步应优先检查优化、表示容量及数据异质性。

### 5.6 V3/V4 能说明什么

- V3 原计划在 Hybrid4 上加入固定 late cosine learning-rate decay，以减轻后期回落；但只完成 4/30 个非候选运行就因任务范围调整而停止，不能用于效果判断。
- V4 保留 Hybrid4+late-decay 候选，并在 SST-2/MRPC/QQP 上完成 45/45
  Stage-A 开发运行，0 failure/INVALID。
- 候选在 7/9 单元达到 C95，但仅 5/9 通过严格门禁。SST-2/352 和
  MRPC/353 未达目标，QQP/352 和 353 有命中后跌破。
- Late decay 对 QQP 有大幅正向终点变化，对 SST-2 略为负向，不是跨任务稳定
  改进。V4 决策为 `NEGATIVE_RESULT`，不进入 Stage B 或 confirmation。

## 6. 原本预期达到的效果是什么

预期效果不是要求压缩方法在所有情况下都超过 dense FedAvg 的最终准确率，而是同时满足以下三层目标。

### 6.1 核心方法目标

在相同任务、相同 seed、相同训练过程和完全相同的 10%/20% serialized wire cap 下，Spectral-A+EF 应稳定优于 Random-A+EF。预注册的实际意义阈值为：SST-2/MNLI 至少提高 0.003，RTE 至少提高 0.01，并且最好在每个 seed 上方向一致。

### 6.2 通信效率目标

压缩方法应在远低于 dense 的通信量下：

- 达到同 seed dense FedAvg 最终性能的至少 95%；
- 每个 seed 都有 `Speedup95 > 1`；
- 固定最后一轮仍保持 `retention >= 0.95`；
- 第一次命中目标后不再跌破；
- TailDrop 不超过任务阈值。

因此，理想效果是“少传很多、较早达到目标、结束时仍保持”，而不仅是某一轮偶然达到高值。

### 6.3 泛化与可信度目标

同一结论应在多个任务、预算和未见 seeds 上成立，并在 official validation 上由冻结方案确认。只有达到这一层，才能把结论写成“稳定方法”；否则只能写成特定任务/预算下的观察。

## 7. 当前证据与预期之间的缺口

| 预期主张 | 现有证据 | 当前状态 |
|---|---|---|
| exact-wire 实现正确、实际节省 80%–90% 通信量 | packet、codec、EF ledger、hash、cap 均通过 | 已支持 |
| 压缩完整 LoRA 可接近 dense 性能 | SST-2 20%、MNLI 10%/20% 有较强结果 | 部分支持，RTE 失败 |
| Spectral-A 稳定优于 Random-A | 仅 MNLI 20% 为一致正向，RTE 10% 一致负向 | 未支持 |
| 固定-A Top-k 性能提升通常来自 EF | 同 wire 的 36 个 EF/no-EF 对为 32 正、2 平、2 负 | 描述性支持；仅限双向 EF 联合效果 |
| 性能提升可以归因于谱子空间 | dense/no-EF 对照均显示强任务与 seed 依赖 | 普遍优势未支持；MNLI 特定条件正向 |
| 达标后稳定保持 | Full-LoRA 仅 7/18；V2 候选最高 4/6；V4 候选 5/9 | 未支持 |
| 优于 FLASC | 评价的模型状态语义不同 | 只能描述，不能归因 |
| 具有统计显著性 | 只有三个正式 seeds | 未支持 |
| 可泛化到其他模型、异质程度和参与率 | 单一 backbone、alpha=0.3、10 客户端全参与 | 未支持 |
| 带来真实系统时延或能耗收益 | 只测应用层 serialized bytes 与单机运行时间 | 未测试 |

## 8. 目前还需要补哪些非实验内容

### 必须补

1. **重写论文主线。** `main.tex` 的 Qwen/Banking77/P2-P3-A2 结果与当前证据不兼容，不能混写。新的主线应围绕 RoBERTa/GLUE、exact-wire、任务依赖与稳定性展开。
2. **锁定唯一核心主张。** 必须在“报告一个完整的负/混合结果”与“继续开发并确认 Hybrid4+decay 新方法”之间做清楚区分。V4 即使成功，也不能自动证明原来的 Spectral-vs-Random 主张。
3. **补齐结果图。** 至少需要 accuracy/score–wire Pareto 图、逐 seed 学习曲线与 C95 目标线、Spectral−Random 配对图、稳定性热图、TailDrop/命中后回落图，以及 residual/support 诊断图。当前工作区没有可用于新故事线的成套论文图。
4. **建立结果版本表。** 正式、supplement、development、smoke 和 superseded 必须在正文或附录中显式标注，避免把 smoke 或 post-hoc 结果当确认结果。
5. **补数据与复现说明。** 从冻结 manifest 自动导出数据集版本、各 split 行数、标签顺序、hash、预处理和 partition 摘要；说明训练 seed 与 partition seed 的关系。
6. **统一比较语义。** FLASC 的 masked-server 指标必须单列；如不补同状态评估，就不能写“显著优于 FLASC”。
7. **补齐本地原始产物。** 联合 CSV 中的正式/补充 `summary_path` 指向远端路径，本地没有对应 108 个 summary；Stability-v2 的逐运行文件也只部分保存在本地。在论文归档前应同步原始 summary、逐轮 JSON、配置和 checkpoint，或提供可验证的不可变归档清单。
8. **固化 V4 可执行源码。** 当前根目录 `src/` 仍是原始/补充版本，V4 方法和 MRPC/QQP 支持保存在 staging/部署归档中。最终复现包应明确唯一入口，并让源码、配置、报告脚本和 hash 对应同一冻结版本。

## 9. 还需要补齐哪些实验

### P0：直接决定论文核心结论

#### 9.1 V4 预注册流程的终止决策

V4 Stage A 已完成 45/45，候选仅 5/9 通过，确定性决策为
`NEGATIVE_RESULT`。按协议接受负结果，不运行 Stage B、confirmation、
补 seed、删任务或临时改超参。

#### 9.2 已完成：fixed-A 机制消融

原计划的 54 个 `Random/Spectral-A dense` 与 no-EF Top-k 运行已经全部完成并
通过审计。它支持“双向 EF 通常有益”，同时否定“谱基本身普遍优于随机基”。
由于 seeds401/402/403 已经暴露，该结论保持 post-hoc 描述性身份。

仍缺少 `FedAvg-LoRA+Top-k（无 EF）` 的 18 个 Full-LoRA 对照；只有补做同源码、
同 seed、同 wire 的 A+B EF/no-EF 配对，才能把 EF 机制结论推广到完整 LoRA。

#### 9.3 让 V4 的对照真正回答原始主张

V4 比较的是 FedAvg、Hybrid4 dense、Hybrid4+EF，以及它们的 decay 版本；没有 Random-A+decay+EF 或 pure Spectral-A+decay+EF。V4 已得到负结果；即使当时通过，也只能证明“Hybrid4+decay 整体包有效”，不能证明“谱基优于随机基”。

如果保留原始核心论点，应在同一批新 seeds、同一 schedule 和同一 wire 下增加：

- Random-A+decay+EF；
- Spectral-A+decay+EF；
- Hybrid4-A+decay+EF。

这些对照只能在新预注册和新 seeds 中作为独立后续研究，不得追加到已终止的 V4。

此外，V4 是在已经观察到 RTE 负结果后定义的新任务套件。RTE 必须继续作为历史负结果和边界报告；若论文仍要提出跨任务普适主张，应通过一份新的预注册协议把 RTE 纳入独立压力测试，而不能用任务范围变更抹去原失败。

#### 9.4 诊断 MNLI 异常 reference

对 seed 403 补做并预注册：标签与客户端 shard 分布、逐轮 loss/accuracy、混淆矩阵、更新范数、分类头行为和重复 fresh seeds。还需要定义 dense reference 的健康门槛及异常 reference 的报告规则。该 seed 不能事后删除，但可与完整结果并列给出敏感性分析。

### P1：使对比和统计结论可信

1. **V4 新任务的完整基线。** MRPC/QQP 目前只有候选与机制对照；若要声称相对现有方法更好，应在相同设置下补 FedAvg-LoRA、FFA-LoRA、LoRA-XS、FLASC-style、Random-A+EF、Spectral-A+EF、Full-LoRA+Top-k+EF 等强基线。
2. **FLASC 同状态评估。** 增加 communicated-global 指标或严格复现其原协议，消除 masked-server 与其他方法的状态语义差异。
3. **增加 fresh seeds。** 当前三个 seeds 只支持描述性配对。seed 数应由预先功效分析决定，并报告配对置信区间、效应量、最差 seed 和多重比较策略。
4. **拆分 EF 方向。** fixed-A 的 no-EF 与 bidirectional EF 已完成；仍需 uplink-only 与 downlink-only EF，才能说明哪一侧真正产生收益。
5. **失败案例分析。** 对未达到 C95、命中后回落和 TailDrop 超限的单元，报告分层 support、residual backlog、更新余弦、类别混淆和客户端差异。

### P2：若要扩大论文结论范围

1. **异质性与参与率：** 至少补 Dirichlet alpha 的多个水平、部分客户端参与和客户端掉线。
2. **预算与 rank 曲线：** 在 10%/20% 之外补 5%/50%，并测试 rank 4/8/16，形成完整 accuracy–wire 边界。
3. **跨模型验证：** 至少增加一个不同 backbone 或模型规模；否则结论必须限定为 RoBERTa-base。
4. **系统实验：** 如果要使用“实际加速”“低延迟”或“节能”等表述，需要真实带宽/时延或网络仿真、端到端时间、CPU/GPU 开销和能耗；当前只能主张序列化字节减少。
5. **测试集或外部验证：** 当前正式结果使用 GLUE validation。可增加官方 test-server 或独立未见测试集，但必须在方法冻结后进行。

## 10. 推荐的最短完成路线

### 路线 A：以当前证据完成一篇诚实、闭合的毕业论文

把主结论改为：exact-wire Top-k+EF 能显著减少联邦 PEFT 通信，在 SST-2/MNLI 的部分条件下保持接近 dense 的效用；fixed-A 消融显示双向 EF 通常有益，但其收益对任务和 seed 敏感；预训练谱子空间没有显示普遍优势，三个稳定化候选也未通过严格开发门槛。补齐正文、图表和复现归档，不再以“稳定正向提升”作为毕业条件。

### 路线 B：继续争取正向方法结论

V4 已按冻结规则终止，不能在当前 seeds 上继续调参。若将来仍要争取正向方法结论，
必须将它定义为新一代、新预注册和新开发 seeds 的独立研究，不能追加到 V4。

无论选择哪条路线，补实验的作用都是检验假设，而不是保证得到预期效果；如果预注册门槛再次失败，应保留并解释负结果。

## 11. 术语与口径

| 术语 | 本文固定含义 |
|---|---|
| wire | 实际序列化应用层字节，包括 FP16 values、uint32 indices、header、dense head、上行和按接收客户端计数的下行；不等于 TCP/TLS/安全聚合总流量 |
| 10%/20% budget | 相对完整 LoRA 单向 dense message 的严格字节 cap，不是坐标保留率 |
| C@95%-Dense | 首次达到同 seed dense FedAvg 最终主指标 95% 时的累计 delivered bytes |
| Speedup95 | dense C95 bytes / 当前方法 C95 bytes；未达到目标时不报告 |
| retention | 当前方法固定最终轮主指标 / 同 seed dense FedAvg 固定最终轮主指标 |
| strict stable | 技术有效、达到 C95、最终仍达标、命中后不再跌破、retention 合格且 TailDrop 合格 |
| formal | 冻结协议、未见正式 seeds、official validation 的可报告证据 |
| development | 用于选方法或解释机制，不可替代确认结果 |
| smoke | 只验证实现链路，不用于比较方法效果 |

## 12. 主要证据文件

- [实验协议](EXPERIMENT_PROTOCOL.md)
- [ACL 论文计划与原始中心问题](ACL_PAPER_PLAN.md)
- [正式结果汇总](results/RESULTS.md)
- [正式结果分析](results/ANALYSIS.md)
- [81-run 原始矩阵](results/roberta_screen/RESULTS.md)
- [27-run Full-LoRA exact-wire 补充实验](results/fedavg_lora_wire/RESULTS.md)
- [Stability-v2 负结果](results/stability_v2/development/STABILITY_REPORT.md)
- [Stability-v4 预注册计划](results/exploration/STABILITY_V4_PLAN.md)
- [V4 启动前验证](results/stability_v4/provenance/VALIDATION.md)
- [V4 Stage-A 稳定性报告](results/stability_v4/development/STABILITY_V4_REPORT.md)
- [V4 确定性决策](results/stability_v4/development/STABILITY_V4_DECISION.json)
- [Fixed-A 机制消融协议](results/exploration/FORMAL_FIXED_A_ABLATION_PLAN.md)
- [Fixed-A 54-run 结果](results/fixed_a_ablation/formal/RESULTS.md)
- [Fixed-A 联合 basis/EF 报告](results/fixed_a_ablation/formal/JOINT_ABLATION_REPORT.md)
- [Fixed-A 联合审计](results/fixed_a_ablation/formal/JOINT_AUDIT.json)
