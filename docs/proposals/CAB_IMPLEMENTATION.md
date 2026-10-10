# CAB（通信感知固定基）实现说明与运行方案

> 日期：2026-10-06。代码已实现，CPU 单元测试通过。**尚未在 GPU 上运行。**
> 理论见 [CLOSED_FORM_NOVELTY.md](CLOSED_FORM_NOVELTY.md) 定理 2；同步规则复用 [DCEF](DCEF_IMPLEMENTATION.md)。

## 1. 方法

| 方法 | 固定基 $A$ | 同步 | 作用 |
|---|---|---|---|
| `cab_a_dcef` | $\mathrm{TopEig}_r(\bar G^\top\bar G-\kappa\,\Sigma_{\mathrm{het}})$ | DCEF | 完整 CAB |
| `cab0_a_dcef` | $\kappa=0$：共识梯度主方向 | DCEF | 关键消融：隔离 $-\kappa\Sigma_{\mathrm{het}}$ 这一新项 |
| `random_a_dcef` / `spectral_a_dcef` | 随机正交基 / 预训练谱基 | DCEF | 现有基线（已在 DCEF 中实现） |

四个 fixed-A 臂共享学习率、DCEF、消息上限和 minibatch 种子，**唯一的差别是 $A$**。

**$\kappa$ 的闭式解。**
$$\kappa=\frac{\gamma\,\omega/N_{\mathrm{eff}}}{2-\gamma(1+\omega/N_{\mathrm{eff}})},\qquad \omega=\frac{d}{K}-1,$$
- $d$ 为 B-only 坐标数（RoBERTa q/v、$r=8$ 时为 147,456）；
- $K$ 为该预算下的下行稀疏坐标数，运行时会核对与压缩器的 $K$ 一致；
- $N_{\mathrm{eff}}=(\sum n_i)^2/\sum n_i^2$；
- $\gamma$ 取自 DCEF 开发阶段的实测值（见第 3 节）；
- `cab.kappa_multiplier` 用于鲁棒性扫描；当 $\gamma$ 超出公式的有效区间时，必须设置 `cab.kappa_override`，否则运行在开始时就会报错。

**构造流程。** 在任何哈希或不变量计算之前完成：
1. 先按 random-A 构造模型作为占位。此时 $B=0$，前向等于原模型。
2. 每个客户端在 eval 模式下，用至多 `gradient_examples` 个样本计算各 q/v 冻结权重的平均梯度 $G_i$。种子与训练 minibatch 种子互不重叠。
3. 每层做秩 $k$（`client_rank`）截断 SVD，打包成一个带 2 的幂缩放的 FP16 dense 包上传。
4. 服务器重建低秩梯度，流式累积 $\bar G$、$\sum p_i^2G_i$、$\sum p_i^2G_i^\top G_i$，得到 $M$，取前 $r$ 个特征向量（符号规范化）。
5. 所有层的基作为一个 FP16 包一次性下发；各方用相同的 CPU float64 QR 重新正交化后写入 `fixed_A`。
6. 基缓存在运行目录的 `cab_basis.pt` 中，并带有指纹校验。断点续跑直接复用缓存，避免 GPU 非确定性改变基的哈希；配置变化时缓存会被拒绝。

**为什么用截断 SVD 而不用随机 sketch。** 在 768×768 合成梯度上做同字节对比（每层每客户端 24 KB）：截断 SVD 的目标值是精确最优的 1.004–1.07 倍，高斯左 sketch 是 1.10–1.25 倍。

## 2. 字节与代价（RoBERTa q/v、$r=k=8$、10 个客户端）

| 项 | 每客户端 | 全体 |
|---|---:|---:|
| 上行：梯度因子 $24\times8\times(768+768)$ 个 FP16 值 + 指数 | 589,839 B | 5.90 MB |
| 下行：基 $24\times8\times768$ 个 FP16 值（发给每个客户端） | 294,925 B | 2.95 MB |
| **一次性 setup 合计** | | **8.85 MB** |

- **占训练字节的比例**：10% 预算下 SST-2 约 25%，RTE/MNLI 约 15%；5% 预算下约 50% / 30%；20% 预算下约 12% / 7.5%。CAB 的 C95 **包含**这部分字节，报告中另列不含 setup 的 `c95_training_bytes`。这是如实计价的结果：CAB 必须靠最终质量和稳定性取胜。
- **计算**：每个客户端约 1024 个样本的一次前向与反向，大约相当于一轮本地训练；另有 240 次 768×768 SVD。
- **显存**：每层 float64 统计量约 28 MB，24 层约 0.7 GB。

**每次运行的 `summary.json` 中 `cab` 一节记录的零训练诊断：**
- $\kappa$、$\gamma$、$N_{\mathrm{eff}}$、$\omega$；
- 上报能量比（截断保留了多少梯度能量）；
- 每层正特征值个数，以及异质与共识的迹之比；
- 与精确最优、谱基、随机占位基、$\kappa=0$ 基的子空间重叠；
- 在精确统计下，所选基、精确最优、谱基、随机基各自的预测目标值 $J(P)/\mathrm{tr}(S)$。

其中谱基与随机基的 $J$ 之差就是预测 P6：不训练即可预测谱基是否优于随机基。

## 3. 运行顺序（服务器）

```bash
python3 -m pytest -q tests/test_cab.py tests/test_dcef.py tests/test_dcef_report.py tests/test_engine.py tests/test_config.py
```
```bash
python3 scripts/preflight.py --config configs/roberta_cab_smoke.yaml
python3 scripts/run_matrix.py --config configs/roberta_cab_smoke.yaml
```

**Smoke 需要检查的内容。**
- 3 个运行全部 COMPLETE；
- CAB 运行的 `final.wire.setup_bytes` 等于 `cab.setup_bytes`；
- `invariants.before.fixed_basis_recipe` 为 `cab_client_svd_fp16_v1`；
- `cab.summary.objective_gap_vs_exact_mean` 接近 1；
- 删掉 checkpoint 之外的文件并续跑时，基被复用（`fixed_basis_hash` 不变）。

**正式开发阶段**依赖 DCEF 开发阶段的 $\gamma$：
```bash
python3 scripts/run_matrix.py --config configs/roberta_dcef_dev.yaml --include-diagnostics
python3 scripts/build_dcef_report.py --config configs/roberta_dcef_dev.yaml
python3 scripts/calibrate_cab_gamma.py --runs docs/results/dcef_dev/DCEF_RUNS.csv
```
把输出的 `gamma` 与 `gamma_source: dcef_dev` 填入 `configs/roberta_cab_dev.yaml`。在 provisional 状态下，配置校验会拒绝启动。然后运行：
```bash
python3 scripts/preflight.py --config configs/roberta_cab_dev.yaml
python3 scripts/run_matrix.py --config configs/roberta_cab_dev.yaml
python3 scripts/build_dcef_report.py --config configs/roberta_cab_dev.yaml
```

- **规模**：开发阶段为 seeds 361–363、train-holdout 评估、3 任务 × 3 预算，共 **117 个运行**，约 10 A800·小时。`--include-diagnostics` 会再加 27 个 `full_lora_dcef` 运行，约 2.3 小时。
- **$\kappa$ 倍率扫描**（0.5×、2×）：复制配置，修改 `kappa_multiplier` 和 `output_dir`。

## 4. 建议在看到结果前定下的判定规则

1. **主要终点**：27 个单元中 `cab_a_dcef` 对 `cab0_a_dcef` 的最终主指标胜负（阈值 SST-2/MNLI 0.003、RTE 0.01）及严格稳定数。理论预测优势随预算减小而增大：若 5% 下“胜 > 负”且总体“胜 ≥ 负”，视为支持 $-\kappa\Sigma_{\mathrm{het}}$ 项。
2. **次要终点**：CAB 及 $\kappa=0$ 相对随机基、谱基的提升（预期较大，但这部分不是新颖性来源）；与 `full_lora_dcef` 的跨族比较，按含 setup 的字节计。
3. **P6**：报告中 $J_{\mathrm{spectral}}-J_{\mathrm{random}}$ 的符号，与实测 `spectral_a_dcef − random_a_dcef` 的符号是否一致。
4. **止损**：若在所有预算下 CAB 与 $\kappa=0$ 都无差别，就不再把 $-\kappa\Sigma_{\mathrm{het}}$ 作为新颖性，只保留“谱基是错误目标的解”这一解释和任务对齐基的工程收益。

## 5. 已知局限

1. **$\kappa$ 的来源**：$\kappa$ 来自无偏压缩器的稳态模型，在 Top-k + DCEF 下只是近似。合成实验中理论 $\kappa$ 接近经验最优，但仍有变差的实例，所以需要倍率扫描。
2. **梯度代理**：用初始点梯度（分类头随机）代理更新方向，与 LoRA-GA、Fed-SB 的假设相同；没有实现分类头预热，也没有训练中刷新基。
3. **采样噪声**：有限样本（每个客户端 1024 条）的梯度噪声会被计入 $\Sigma_{\mathrm{het}}$。随机梯度确实也会造成压缩噪声，但两者并不等价。
4. **扩展性**：统计量按层保存 $m\times n$ 与 $n\times n$ 的 float64 矩阵，用于 7B 级 LLM 时需要改为逐层流式计算。
5. **秩约束**：`client_rank` 目前要求不小于 LoRA 秩，这是保守设置，不是理论要求。
