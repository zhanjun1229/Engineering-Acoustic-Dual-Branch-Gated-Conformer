# 本地验证记录

版本：0.1.2。验证日期：2026-10-19（Asia/Shanghai）。

## 验证环境

- Windows，Python 3.13.9。
- 项目内虚拟环境共享主机已有 site-packages；没有在全新环境中重新下载全部依赖。
- PyTorch 2.11.0+cu128，其他直接依赖版本见 `requirements.txt`。
- 模型测试显式使用 CPU，特征测试使用合成音频，不需要 MIMII 数据。

## 检查范围

| 检查 | 内容 |
| --- | --- |
| Ruff | 导入、所选静态检查规则和格式 |
| pytest | 9项合成输入与实现一致性测试 |
| 参考一致性 | 71个函数、类和常量的执行 AST；忽略注释和文档字符串 |
| 命令行 | 安装后的 `conformer-train --help`、模块入口和脚本入口 |
| 打包 | wheel、源码发行包和上传 ZIP 的内容与完整性 |

测试涵盖实验默认值、文件级分层划分、Acoustic156/IAPD26 维度与有限值、模型前向、Gate 归一化、state_dict 保存加载、指标边界、文件级均值聚合及早停快照。

## 本地检查命令

在包含 `pyproject.toml` 的项目目录安装开发依赖后运行：

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest
python -m build --no-isolation
conformer-train --help
```

受限执行环境中，如系统临时目录不可写，可将临时目录和 Numba 缓存放在项目内；普通本地环境通常无需此设置：

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null
$env:TEMP = (Resolve-Path .tmp).Path
$env:TMP = $env:TEMP
$env:NUMBA_CACHE_DIR = Join-Path $env:TEMP 'numba'
python -m pytest -q --basetemp=.tmp/pytest
```

## 验证边界

没有执行完整 MIMII 训练、论文结果复现、GPU 训练或跨平台安装验证。合成输入测试通过不代表论文全部实验已经复现，也不能保证不同硬件或依赖环境中的训练结果逐位一致。

运行时失败文件处理、缓存复用和指标字段定义见 README。模型结构、特征计算和实验参数未作功能性修改。
