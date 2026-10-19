# Dual-Conformer with Dynamic Gate

机器声音监督二分类主方法实现：Acoustic156 与 IAPD26 分别进入独立的 Conformer 编码器，池化后通过样本级 Softmax Gate 加权，再拼接分类。

本项目公开主方法所需的特征提取、模型训练和文件级评估，不包含其他方法的对比实现或超参数搜索程序。当前数据读取与实验协议针对 MIMII 数据集，项目更名不改变数据支持范围。当前版本号为 `0.1.2`；实现范围见 [docs/PACKAGING.md](docs/PACKAGING.md)。

## 项目结构

```text
.
├── src/dual_conformer_gate/
│   ├── cli.py          # 命令行参数与入口
│   ├── config.py       # 特征与实验常量
│   ├── features.py     # Acoustic156 / IAPD26 提取与分段
│   ├── dataset.py      # 数据发现、文件级划分、缓存和 DataLoader
│   ├── model.py        # 双分支 Conformer 与动态 Gate
│   ├── metrics.py      # 文件级聚合和评估指标
│   ├── trainer.py      # 早停、模型保存、训练及批量实验
│   └── utils.py        # 随机种子、日志和结果汇总
├── tests/              # 实现一致性与合成数据检查
├── docs/               # 实现范围与验证记录
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── train.py
```

## 安装

推荐创建独立环境。验证环境为 Python 3.13.9；依赖版本见 `requirements.txt`。当前依赖组合要求 Python 3.11 及以上，未在所有 Python 版本和系统上验证。首次使用优先选择 Python 3.13。

以下命令均在包含 `pyproject.toml` 的仓库根目录执行。

```bash
python -m venv .venv
```

Windows PowerShell 激活：

```powershell
.venv\Scripts\Activate.ps1
```

Linux / macOS 激活：

```bash
source .venv/bin/activate
```

在上述项目目录安装：

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`.[dev]` 同时安装测试、格式检查和打包工具；只训练可使用 `python -m pip install -e .`。

需要 GPU 时，先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装与驱动和系统匹配的 PyTorch，再安装本项目。`requirements.txt` 中的 `torch==2.11.0` 对应本地验证使用的 `2.11.0+cu128`，单独安装本项目不保证得到相同 CUDA wheel。

这是一份本地验证使用的**直接依赖版本清单**，不是所有间接依赖的完整锁文件，也未作为论文结果的完整训练环境验证。确认自己的环境可完成实验后，可另存完整版本记录：

```bash
python -m pip freeze > environment-resolved.txt
```

## 数据准备

原始 WAV、预提取特征和训练权重未随仓库提供。自行获取 MIMII 数据，并按以下可识别结构整理；数据获取和使用条件请查阅 [MIMII 官方项目](https://github.com/MIMII-hitachi/mimii_baseline)。本项目不会自动下载数据。

```text
data/
├── -6_dB_fan/
│   ├── id_00/
│   │   ├── normal/*.wav
│   │   └── abnormal/*.wav
│   ├── id_02/...
│   ├── id_04/...
│   └── id_06/...
├── 0_dB_fan/...
├── 6_dB_fan/...
├── -6_dB_pump/...
└── ...
```

默认机器为 `fan`、`pump`、`slider`、`valve`；噪声键 `m6`、`0`、`6` 分别表示 -6、0、+6 dB。每个组合必须同时含有正常和异常文件。机器 ID 默认为 `id_00,id_02,id_04,id_06`，可用 `--machine_ids` 指定。

可选预提取声学特征目录：

```text
features/fan/m6/id_00/normal/<WAV文件名去扩展名>.npz
```

每个 NPZ 需包含 `mfcc39`、`lfcc39`、`cqcc39`、`gfcc39` 四个二维数组，形状均为 `[帧数, 39]`，且编码定义应与本项目一致。IAPD26 仍从 WAV 提取。未提供 `--feature_root` 或某个 NPZ 不可用时，从 WAV 计算声学特征。

## 运行

检查命令行参数：

```bash
conformer-train --help
```

先运行一组实验：

```bash
conformer-train --data_root ./data --machine fan --noise m6 --output_root ./results --use_cache
```

运行全部 12 组：

```bash
conformer-train --data_root ./data --output_root ./results --use_cache
```

使用预提取特征：

```bash
conformer-train --data_root ./data --feature_root ./features --machine fan --noise m6 --use_cache
```

也可以在安装后通过 `python -m dual_conformer_gate` 或 `python train.py` 使用同样参数。Windows 路径包含空格时请加引号。

`--data_root` 必填；`--feature_root` 默认为空；`--output_root` 默认为 `./results`。使用预提取特征时应明确传入其根目录，避免无意切换特征来源。

## 实验协议与固定参数

本实验使用正常标签 0、异常标签 1，训练集包含两类数据，属于监督分类。四类机器与三种噪声组成 12 个独立域内实验，不进行跨噪声迁移或机器 ID 留出评估。

先按原始 WAV 文件和类别分层划分，再生成重叠片段：20% 文件用于 Test，其余文件中 20% 用于 Dev，整体比例约为 Train / Dev / Test = 64% / 16% / 20%。有限样本下数量受取整影响。默认同一机器的多个 ID 会混合后进行文件级划分。

| 参数 | 默认实验设置 |
| --- | --- |
| Acoustic 分支 | MFCC39 + LFCC39 + CQCC39 + GFCC39，共 156 维 |
| 工业特征分支 | IAPD26 |
| 声学前端 | MFCC / LFCC / GFCC 使用 spafe；CQCC 使用 0.97 预加重、CQT、log1p、DCT |
| 采样率 | 16000 Hz |
| FFT / 窗长 / 帧移 | 512 / 400 / 160 |
| 分段长度 / 步长 | 128 / 64 帧 |
| d_model / heads / blocks / ff_mult | 64 / 4 / 2 / 2 |
| 编码器 dropout | 0.10654172080387744 |
| 分类器 | 128 → 256 → 128 → 1，dropout 为 0.5、0.3 |
| batch size | 64 |
| Adam 学习率 | 0.0005184655055267425 |
| 最大 epoch / 早停 patience | 50 / 10 |
| 梯度裁剪 | 3.0 |
| seed / num_workers | 42 / 0 |
| IAPD CMVN epsilon | 1e-5 |
| 选择最佳模型 | 文件级 Dev EER 最小值 |
| 文件级聚合 / 分类阈值 | 片段概率均值 / 0.5 |

本版本使用表中的固定超参数，不执行超参数搜索。固定种子及 cuDNN 设置不保证不同硬件、依赖或 CUDA 版本间逐位一致。

### 指标定义

输出包含文件级 F1、ROC AUC、EER、混淆矩阵及其他统计。EER 取 ROC 上 FPR 与 FNR 最接近的离散点，再取两者均值；不做插值。只有单一类别时，EER 和下述代价返回 1.0。

结果字段 `min_tDCF` 的实际计算公式是：

```text
min_threshold [0.75 × FNR(threshold) + 0.25 × FPR(threshold)]
```

请按这个具体定义解释结果；本实现没有串联系统的额外代价项。Gate 统计沿用片段级权重的均值和标准差。

## 输出文件

```text
results/
├── summary_4machines_3noises.csv
├── _shared_feature_cache/<feature_version>/<machine>/<noise>/
└── fan_m6_mfcc_lfcc_cqcc_gfcc_iapd26_gate_s42/
    ├── run.log
    ├── best_model.pt
    ├── completed.json
    ├── training_curve.png
    ├── splits/split.npz
    └── outputs_significance/
        ├── dev_best.npz
        └── test.npz
```

实验失败时会保存 `failed.json` 并继续后续组合。`completed.json` 存在时默认跳过该组合，使用 `--force` 可重跑。

## 运行行为与复现注意事项

- **修改实验配置后使用新的输出目录。** 已完成实验的目录名只包含机器、噪声和 seed，学习率等参数变化不会触发自动重跑；总表也按机器、噪声覆盖行。建议每个配置或 seed 使用独立的输出根目录。
- **修改特征设置后使用新的缓存目录。** 缓存有 `FEATURE_VERSION` 检查，但没有覆盖所有配置参数或原始音频内容。`--force` 重跑训练不会清除缓存。
- 特征读取失败会警告并跳过文件；划分清单和结果中的文件数量来自最初划分，可能多于实际参与评估的文件。正式复现实验需核对日志中失败文件和有效样本情况。
- 缓存和部分输出使用 NumPy 对象数组；仅对自己生成且可信的文件使用 `allow_pickle=True`。
- `--no_plot` 是保留的兼容参数；传入后仍保存训练曲线，始终不弹出图形窗口。
- 批量入口捕获单组错误并继续，因此进程成功退出不代表所有实验成功；请检查总表的 `status`。
- Dataset 会将提取后的全部片段装入内存。首次计算特征可能较慢，也需要足够内存和磁盘空间。

## 开发与验证

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest
python -m build --no-isolation
```

修改格式可执行 `python -m ruff format .`。上述检查可以在本地手动运行，不依赖 GitHub Actions。测试使用合成数据，不运行完整 MIMII 训练。

测试覆盖参考实现中 71 个函数、类和常量的执行 AST 指纹、命令行默认值、文件划分无交集、合成音频特征维度、Gate 归一化、权重加载和早停。执行 AST 比较忽略注释和文档字符串，仍检查计算表达式、参数、默认值及常量。后续有意修改研究实现时，应审查变更并更新基线说明。合成数据测试不能替代论文结果复现。

实现范围见 [docs/PACKAGING.md](docs/PACKAGING.md)，本地验证记录见 [docs/VALIDATION.md](docs/VALIDATION.md)。

## 许可证

本项目采用 [MIT License](LICENSE)。第三方软件与数据遵循各自的许可条件。

