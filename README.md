# fedpeft：联邦大模型参数高效微调的轻量化同步

面向联邦 LoRA 微调的实验代码库（硕士论文《面向联邦大语言模型参数高效微调的轻量化同步方法研究》）。

核心方法是**曲率感知的固定共享基**：所有客户端共享一个固定的 LoRA 矩阵 A，每轮只训练、只同步 B。这样聚合是精确的，可以叠加 Top-k + 误差反馈压缩，也兼容安全聚合。A 由一个优化目标的闭式解给出，不是人为指定的：

- 只看一步下降的一阶项，即"捕获的梯度能量"（LoRA-GA 式的固定基、EVA），会选中激活中能量极高的少数方向。投影能量是随机基的 74–160 倍，联邦训练因此停在多数类上（RoBERTa 和 Qwen2.5 上都已复现）。
- 把曲率项加回来（Gauss-Newton / K-FAC 近似下的最优一步下降），约束变为 $A\Sigma_xA^\top=c^2I$，闭式解是 $(\bar G^\top\bar G,\ \Sigma_x)$ 的前 r 个广义特征向量（代码里的 `gev0`）。
- 协议：客户端上传截断梯度因子和输入二阶矩草图，服务器求解后广播 A；再加一轮很小的 Gram 归一化，让归一化约束精确成立。所有通信字节都计入 setup bytes。

完整的研究过程、负结果和设计转折见 [`docs/proposals/RESEARCH_DIRECTIONS.md`](docs/proposals/RESEARCH_DIRECTIONS.md)。

## 主要结果（确认阶段，官方验证集，种子 1001–1003）

RoBERTa-base，10 个客户端，Dirichlet α=0.3，双向 Top-k + 误差反馈（预算按真实字节计）。下表是最终分数的 3 种子均值。

| 方法 | MNLI 5% / 10% / 20% | QNLI 5% / 10% / 20% | SST-2 5% / 10% / 20% |
|---|---|---|---|
| random 固定 A | 0.749 / 0.766 / 0.777 | 0.780 / 0.801 / 0.804 | 0.708 / 0.821 / 0.849 |
| spectral 固定 A | 0.751 / 0.770 / 0.778 | 0.801 / 0.812 / 0.825 | 0.628 / 0.676 / 0.786 |
| cab0（梯度能量基） | 0.472 / 0.613 / 0.453 | 0.650 / 0.770 / 0.775 | 0.811 / 0.856 / 0.876 |
| **gev0（本方法）** | **0.793 / 0.806 / 0.809** | **0.835 / 0.842 / 0.849** | **0.833 / 0.872 / 0.886** |

- gev0 对 random：27 个配对全胜，p = 1.5e-8（精确符号检验）。
- 不压缩时，gev0 固定 A 只训练一半参数，和完整 LoRA FedAvg 打平（9 个配对 4 胜 / 1 平 / 4 负），达到 C95 用的字节少 1.5–3 倍。
- 完整表格（含 C95 字节与消融）：[`docs/results/confirm/CONFIRM_SUMMARY.md`](docs/results/confirm/CONFIRM_SUMMARY.md)。LLM 零训练检查：[`docs/results/llm_check/`](docs/results/llm_check/)。

与已发表的联邦 LoRA 方法比较（同一协议；每个方法、包括本方法，都用同一套开发集扫描选出的学习率；[`docs/results/baselines/`](docs/results/baselines/BASELINES_SUMMARY_SYMMETRIC_LR.md)）：

| gev0 − 基线（压缩，27 对） | 平均差 | 胜 / 平 / 负 | p |
|---|---:|---:|---:|
| FFA-LoRA（ICLR'24）+ Top-k/EF | +0.063 | 27 / 0 / 0 | 1.5e-8 |
| FedIT + Top-k/EF | +0.042 | 22 / 4 / 1 | 5.7e-6 |
| RoLoRA（NeurIPS'25）+ Top-k/EF | +0.048 | 26 / 0 / 1 | 4.2e-7 |
| FedSVD（NeurIPS'25）+ Top-k/EF | +0.028 | 23 / 2 / 2 | 1.9e-5 |
| Fed-SB（TMLR'26，秩按预算取） | +0.161 | 26 / 1 / 0 | 3.0e-8 |

- 压缩时 gev0 是唯一在所有任务、预算、种子下都达到 C95 的方法（27/27；次好的 FedSVD 22/27）。
- 不压缩时 gev0 与 FedIT、FedEx-LoRA（ACL'25）精度打平，达到 C95 的字节在三个任务上都是最少的：比 FedIT 少 2.2–3.1 倍，比 FedEx-LoRA 少 10–18 倍。FLoRA（NeurIPS'24）和 FedEx-LoRA 为了精确聚合，每轮通信是 FedIT 的 5.5 倍。
- Fed-SB 的固定基来自梯度奇异向量，和 cab0 一样在 MNLI / QNLI 上明显退化。
- gev0 的学习率扫描只改变了 SST-2（3e-4 → 1e-3）。沿用随机基学习率时，SST-2 5% / 10% 下 FedSVD 更好；扫描后 SST-2 上 gev0 为 0.870 / 0.905 / 0.910，5% 与 FedSVD 持平，10% / 20% 更好。

Qwen2.5-0.5B（decoder，MNLI，同样的 FL 设定，gev0 用低成本协议，建基 9.3 MB；[`docs/results/qwen_confirm/`](docs/results/qwen_confirm/)）：

| 方法 | 5% | 10% |
|---|---|---|
| 完整 LoRA FedAvg（不压缩） | 0.764 | 0.764 |
| random 固定 A | 0.757 | 0.757 |
| cab0（梯度能量基） | 0.636 | 0.588 |
| **gev0** | **0.754** | **0.758** |

- 梯度能量基在 LLM 上同样退化：cab0 对 random 6 个配对全负（−0.145，p=0.031）。
- gev0 修复了退化，但和 random 打平（3 胜 / 3 负）。在这个模型上，随机固定 A 本身已接近完整 LoRA，固定基的选择没有多少提升空间。

## 目录

```
src/fedpeft/      FL 引擎、适配器（LoRA / 固定 A）、压缩与误差反馈、字节计数、建基协议（cab.py）
configs/          实验配置（见下表）
scripts/          单次运行、并行调度、报告与汇总、配置生成、诊断脚本
  server/         服务器上实际使用的启动脚本（tmux 流水线）
  proposals/      零训练诊断与合成检查
tests/            单元测试（CPU 即可运行）
docs/             研究文档；docs/results/ 只保留精简汇总，原始运行留在服务器
```

| 配置 | 用途 |
|---|---|
| `configs/roberta_confirm_main.yaml`, `roberta_confirm_eva.yaml` | 确认阶段（论文数字来源） |
| `configs/roberta_gev_dev.yaml`, `roberta_qnli_dev.yaml` | 开发阶段（train-holdout 评估） |
| `configs/roberta_gev_probe.yaml`, `roberta_gev_oracle.yaml`, `roberta_gev_cost_v*.yaml`, `roberta_gev_checks.yaml` | 探针、oracle、建基成本迭代、EVA / 不压缩对照 |
| `configs/qwen05b_*.yaml` | Qwen2.5-0.5B（decoder，最后一个 token 接分类头） |
| `configs/roberta_cab_ef_dev.yaml`, `roberta_dcef_*.yaml`, `roberta_consensus_*.yaml` | 早期方向（CAB、DCEF、共识测量），作为负结果保留 |

## 方法与关键配置项

方法名（`matrix.dense_methods` / `matrix.sparse_methods`）：

| 方法 | 含义 |
|---|---|
| `fedavg_lora` | 完整 LoRA（A、B 都训练），不压缩，作为 C95 参考 |
| `random_a_ef` / `random_a_dense` | 随机正交固定 A（FFA-LoRA 式），压缩 / 不压缩 |
| `spectral_a_ef` | 预训练权重右奇异向量作为固定 A |
| `cab0_a_ef` | 梯度能量基：$\bar G^\top\bar G$ 的前 r 个特征向量 |
| `eva_a_ef` | EVA 式：输入二阶矩的前 r 个特征向量 |
| `cab0wh_a_ef` | cab0 的子空间，在子空间内白化（消融） |
| **`gev0_a_ef` / `gev0_a_dense`** | **本方法**：广义特征向量 + 曲率归一化，压缩 / 不压缩 |
| `full_lora_ef` | FedIT（完整 LoRA）+ Top-k + 误差反馈 |
| `flora_dense` | FLoRA（NeurIPS 2024）：堆叠聚合，每轮合并进底座权重 |
| `fedex_dense` | FedEx-LoRA（ACL 2025）：平均 A、B，残差合并进底座权重 |
| `rolora_dense` / `rolora_ef` | RoLoRA（NeurIPS 2025）：每轮只训练、只发送 B 或 A，交替 |
| `fedsvd_dense` / `fedsvd_ef` | FedSVD（NeurIPS 2025）：每轮对 BA 做 SVD 重新选 A（子空间实际不变） |
| `fedsb` | Fed-SB（TMLR 2026）：梯度符号 SVD 得到固定 B、A，只训练 r×r 的 R；秩按预算取最大 |

基线的实现与字节口径见 [`src/fedpeft/baselines.py`](src/fedpeft/baselines.py) 的模块说明。

`cab:` 段控制建基协议：

| 配置项 | 含义 |
|---|---|
| `client_rank` | 客户端梯度截断秩 |
| `activation_rank` | 输入二阶矩草图的秩（再加对角余项） |
| `factor_dtype`, `basis_dtype` | 梯度因子 / 基广播的编码，`float16` 或 `int8` |
| `basis_refine: gram` | 增加 Gram 归一化轮，使 $A\Sigma_xA^\top=c^2I$ 精确成立 |
| `share_basis: true` | 读同一输入的模块（同层 q/v）共享一个基，广播量减半 |

确认阶段使用的协议是：`client_rank 4`，`activation_rank 8`，FP16。

## 运行

依赖：Python ≥ 3.10，`pip install -r requirements.txt`（torch 2.8.0、transformers 4.56.2、datasets 3.6.0）。GLUE 数据用 `datasets.load_dataset("nyu-mll/glue", task).save_to_disk(...)` 保存，模型放在本地目录，路径都写在配置里（`model.path`、`data.root` 等，需要按自己的环境修改）。

```bash
export PYTHONPATH=src

# 单元测试
python -m unittest discover -s tests

# 单次运行（可断点续跑；结果写到配置中的 output_dir）
python scripts/run_experiment.py --config configs/roberta_confirm_main.yaml \
  --task mnli --method gev0_a_ef --seed 1001 --wire-ratio 0.10

# 整个配置矩阵并行运行（已完成的运行会被跳过）
python scripts/run_parallel.py --config configs/roberta_confirm_main.yaml --workers 4 --threads-per-run 1

# 报告与汇总
python scripts/build_dcef_report.py --config configs/roberta_confirm_main.yaml
python scripts/summarize_confirm.py <results_root>   # 含 confirm_main/ 与 confirm_eva/
```

常用辅助脚本：

| 脚本 | 用途 |
|---|---|
| `scripts/make_confirm_configs.py` | 生成确认阶段配置 |
| `scripts/make_qwen_main_config.py` | Qwen：按基线扫描结果选学习率并生成主对比配置 |
| `scripts/proposals/cab_basis_anisotropy_check.py` | RoBERTa：各种基的平均方向占比 / 能量倍数 / 可分性 |
| `scripts/proposals/llm_anisotropy_check.py` | LLM（Qwen2.5 等）零训练检查 |
| `scripts/proposals/basis_install_dry.py` | 只走建基流程，不训练，测量建基字节与归一化质量 |

## 可复现性约定

- 每个运行都会写原子的 `summary.json`、逐轮 `round-NNN.json` 和断点文件，并记录配置哈希、源码哈希（`src/**/*.py`）、数据、模型和划分的指纹。
- 不同实验族各自使用独立的代码目录，保证同一族内的源码哈希一致。
- 学习率只对基线调（random 固定 A 和不压缩 FedAvg），选出的值所有固定 A 方法共用；本方法不单独调参。
- 确认阶段的种子、学习率、训练轮次和建基协议，都在开发阶段固定下来。

## 历史阶段

早期的稳定性协议实验（screen、stability v1–v4）、DCEF 和 CAB 的说明，归档在 [`docs/LEGACY_STABILITY_STUDY.md`](docs/LEGACY_STABILITY_STUDY.md) 和 `docs/proposals/` 下。相关代码路径保留，测试仍然覆盖。
