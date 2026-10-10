# DCEF 实现说明与运行方案

> 日期：2026-10-06。代码已实现，CPU 单元测试通过。**尚未在 GPU 上运行**：本地没有 CUDA、模型和数据，`run_experiment` 只能在服务器上执行。
> 推导与动机见 [CLOSED_FORM_NOVELTY.md](CLOSED_FORM_NOVELTY.md)。

## 1. 新增方法

| 方法 | 客户端 | 服务器 | 模型构造（逐位相同） | 作用 |
|---|---|---|---|---|
| `full_lora_dcef` | 从 $x+e_i$ 开始本地训练 | $v=g+(1-\bar\gamma)e_s$ | `full_lora_ef` | 完整方法 |
| `full_lora_dcef_client` | 从 $x+e_i$ 开始本地训练 | 普通 EF：$v=g+e_s$ | `full_lora_ef` | 消融：只有客户端这一半 |
| `random_a_dcef` | 同上 | 同 `full_lora_dcef` | `random_a_ef` | fixed-A 对照（延迟只有 Full LoRA 的一半） |
| `spectral_a_dcef` | 同上 | 同上 | `spectral_a_ef` | 留给后续 CAB 实验 |

每对方法共享模型构造、初始权重、划分、minibatch 种子（`seed·1000003 + round·10007 + client`）、学习率和消息上限，**唯一的差别是同步规则**。

**字节口径。**
- 带服务器折扣的方法，每条上行消息多带一个 FP16 控制值 $\hat\gamma_i$，用一个 15 B 的 dense 包传输，计入上限。
- 为此上行稀疏预算从 $K$ 降为 $K_{\uparrow}=\lfloor(\text{adapter\_cap}-15-21)/6\rfloor$，最多少 3 个坐标（不到 $K$ 的 0.04%）；下行仍为 $K$。
- 这对 DCEF 是保守的，两个 $K$ 都写入 `summary.json` 的 `dcef` 与 `wire_budget` 字段。

**$\hat\gamma_i$ 的估计。** 客户端用本地完整向量做割线最小二乘：
$$\hat\gamma_i=-\frac{\sum\langle d_{t+1}-d_t,\ s_{t+1}-s_t\rangle}{\sum\lVert s_{t+1}-s_t\rVert^2},$$
其中 $s$ 为本轮起点，$d$ 为本地位移。结果截断到 $[0,1]$；第一轮取 0，即退化为普通 EF。合成实验表明，在噪声下该估计会偏大（真值 0.57 时为 0.58–0.74），但对最终误差几乎没有影响；工具变量版本更差，所以没有采用。

**审计。**
- EF 守恒账本对 DCEF 同样成立：客户端输入取本地位移，服务器输入取 $g-\bar\gamma e_s$，两者都由测试验证闭合。
- checkpoint 新增 `dcef_state`（每个客户端的上一轮起点与位移，以及累计的分子分母），断点续跑后估计连续。
- 每轮 JSON 的 `diagnostics.dcef` 记录：
  - `gamma_hat_by_client`、`gamma_bar`、`server_residual_retention`(= $1-\bar\gamma$)；
  - `virtual_offset_norm_mean/max`（$\lVert e_i\rVert$）；
  - `uplink_sparse_k`。
- Full LoRA 变体沿用 `adapter_factor_audit`，记录 A/B 两个因子各被选中多少坐标。

## 2. 改动的文件

| 文件 | 改动 |
|---|---|
| `src/fedpeft/dcef.py`（新） | 方法集合、构造映射、控制值编解码、`SecantGammaEstimator`、虚拟起点与服务器输入两个纯函数 |
| `src/fedpeft/engine.py` | 客户端起点、上行压缩器与控制包、服务器折扣与账本、每轮诊断、checkpoint、summary。**已有方法走的代码路径与改动前逐位相同** |
| `src/fedpeft/config.py` | 注册 4 个方法；DCEF 与对应 EF 方法必须共享学习率（fixed-A 族与 full-LoRA 族） |
| `configs/roberta_dcef_smoke.yaml`、`configs/roberta_dcef_dev.yaml`（新） | 见第 3 节 |
| `scripts/build_dcef_report.py`（新） | 技术审计、稳定性终点、配对差异、机制指数 $\bar\gamma\cdot d/K$ |
| `tests/test_dcef.py`、`tests/test_dcef_report.py`（新） | 15 个测试，其中协议测试经过真实的 wire 编解码与 Top-k 压缩器 |

**注意 `source_hash`。** 新增 `dcef.py` 会改变 `src/` 的哈希。若在同一目录用 `build_report.py` 重建旧的 81/27-run 报告，会报 source 不一致。这与 V4 的情况相同，建议在 staging 副本中部署，或在旧报告旁注明 source delta。

**本地测试结果。**
- 除 `tests/test_report.py` 外全部 148 个测试通过。该模块在本机 Python 3.9 下因 `X | None` 语法无法导入，与本次改动无关。
- 本机的 pandas/numpy 二进制不兼容，测试时用了一个仅在 `/tmp` 的 shim，未写入仓库。
- 服务器使用 Python 3.10 以上，正常运行 `pytest` 即可。

## 3. 运行方案（服务器）

```bash
export PYTHONPATH="$PWD/src:$PWD/.compat:$PWD/.deps"
python3 -m pytest -q tests/test_dcef.py tests/test_dcef_report.py tests/test_engine.py tests/test_config.py
```
```bash
python3 scripts/preflight.py --config configs/roberta_dcef_smoke.yaml
python3 scripts/run_matrix.py --config configs/roberta_dcef_smoke.yaml --include-diagnostics
```

**Smoke 需要检查的内容。**
- 5 个运行全部 COMPLETE；
- `invariants.wire_cap_respected` 为 true，`ef_conservation_max` < 1e-6；
- DCEF 运行的 `dcef.uplink_sparse_k` ≤ `downlink_sparse_k`；
- 第 2、3 轮的 `gamma_bar` 非空且落在 $[0,1]$；
- 中途中断后能够续跑。

```bash
python3 scripts/preflight.py --config configs/roberta_dcef_dev.yaml
python3 scripts/run_matrix.py --config configs/roberta_dcef_dev.yaml
python3 scripts/build_dcef_report.py --config configs/roberta_dcef_dev.yaml
```

**开发阶段设置。**
- seeds 361–363（未曾使用过），train-holdout 评估。
- 任务 SST-2 / RTE / MNLI，预算 5% / 10% / 20%。
- 方法：`fedavg_lora`（dense 参照）、`full_lora_ef`、`full_lora_dcef`、`full_lora_dcef_client`，共 **90 个运行**。按现有单次耗时（SST-2 约 190 s、RTE 约 100 s、MNLI 约 640 s）估算约 **7.8 A800·小时**。
- 加 `--include-diagnostics` 会再跑 fixed-A 的 54 个运行，约 4.7 小时。
- seeds 371–373 预留给冻结之后的 official-validation 确认。

## 4. 建议在看到结果前定下的判定规则

以下是建议，可以修改，但应在启动开发矩阵之前写定。

1. **主要终点**：27 个匹配单元（3 任务 × 3 seed × 3 预算）中，`full_lora_dcef` 与 `full_lora_ef` 的严格稳定数之差。差值 ≥ 3，且按预注册阈值（SST-2/MNLI 0.003，RTE 0.01）统计的最终主指标“胜 ≥ 负”，视为支持。
2. **机制检验**：
   - 标准 EF 的命中后违约总数应满足 5% ≥ 10% ≥ 20%；
   - DCEF 相对 EF 的优势在 5% 时应大于 20% 时；
   - 报告中的 $\bar\gamma\cdot d/K$ 越大，EF 越差。
3. **消融**：`full_lora_dcef` 对 `full_lora_dcef_client` 的胜负，用来回答服务器折扣是否必要。合成实验预测：同质时两者相同，异质时完整方法更好。
4. **止损**：若严格稳定数没有提高，且最终分数胜 ≤ 负，就按负结果记录，不在这三个 seed 上继续调参。
5. **已知的合理例外**：本地每轮推进很小（$\gamma d/K\ll1$）时，标准 EF 的重复累加相当于更大的步长，可能在固定轮数内略快于 DCEF。DCEF 跟随的是 dense FedAvg 轨迹，它的收益应当体现在稳定性和保持率上，不一定体现在收敛速度上。
