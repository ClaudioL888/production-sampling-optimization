# 生产决策与抽样检验参数优化

配套报告：**《基于贝叶斯抽样检测与库存反馈的生产决策优化模型》**。

围绕零配件抽样接收、生产检测与拆解、回收库存反馈等问题，归档原有 Python 分析代码、数值结果、可视化图表、报告及汇报材料。

报告原署名：软件2304 顾雨阳、伍俊宇、李佳融。仓库用于团队项目材料整理，不据此划分成员的个人贡献；未添加竞赛名称、奖项或发表状态。

## 内容与方法

- 抽样检验：单次/双重抽样、接收与拒收概率、期望抽样量与效度指标。
- 参数搜索：`optimize1_3.py` 实际使用 SciPy 差分进化算法；`optimize1.py` 为 scikit-opt 遗传算法对照版本。
- 生产仿真：蒙特卡罗抽样、检测与拆解策略、库存回收反馈、遗传算法策略搜索。
- 扩展分析：贝叶斯双重抽样、不同参数和生产情境的敏感性分析。
- 多工序动态规划：完整报告中包含相关模型与代码段；当前未单独整理为可运行模块。

## 目录

| 路径 | 内容 |
| --- | --- |
| `src/efficacy_function.py` | 单次抽样效度计算与绘图 |
| `src/optimize1_3.py` | 双重抽样参数优化（差分进化）及结果绘图 |
| `src/optimize1.py` | 双重抽样参数优化（遗传算法对照） |
| `src/v_optimizer.py` | 权衡效度与样本量的评价指标分析 |
| `src/21.py` | 原始生产仿真实验脚本 |
| `src/42.py` | 包含贝叶斯抽样的生产决策实验脚本 |
| `src/optimized_main_with_parallel_sensitivity.py` | 生产策略与敏感性分析原始并行版本 |
| `src/22.py` | 敏感性分析结果汇总脚本 |
| `src/1.py` | 六种生产情境参数表生成 |
| `results/` | 已有 CSV 结果，包括抽样优化结果与敏感性分析结果 |
| `figures/` | 已有分析图表 |
| `data/` | 生产情境参数表 |
| `paper/document2.pdf` | 原始完整报告 |
| `paper/document2.tex` | 报告 LaTeX 源文件及其引用图片 |
| `docs/` | 汇报 PPT、原始代码说明和归档说明 |
| `SOURCE_MANIFEST.json` | 原始文件相对路径、修改时间与 SHA-256 校验清单 |

## 环境

建议新建 Python 3.11 或 3.12 环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

生产仿真另需 PyTorch：

```powershell
python -m pip install -r requirements-production.txt
```

PyTorch 默认可使用 CPU；需要 GPU 时请按 PyTorch 官方安装说明选择与本机 CUDA 匹配的版本。本次未恢复原项目依赖锁文件，requirements 是根据导入项整理的依赖清单。

## 运行

从仓库根目录运行；新增入口会将工作目录放在 `runs/<任务>/`，并复制必要的已有 CSV，避免覆盖归档结果。

```powershell
# 基于已有 CSV 重新绘制抽样优化图，不重新运行参数搜索
python run.py plot

# 单次抽样效度计算与绘图
python run.py efficacy

# 综合评价指标分析
python run.py metric

# 汇总已有敏感性分析 CSV
python run.py summarize

# 双重抽样参数搜索，计算量较大
python run.py optimize
```

`run.py` 默认使用非交互绘图后端 Agg。原始脚本也可自行在准备好输入 CSV 的工作目录中运行。

生产仿真的原始入口为 `src/21.py`、`src/42.py` 和 `src/optimized_main_with_parallel_sensitivity.py`。**并行敏感性分析版本会在多个线程间共享全局情境参数；参数隔离尚未修正，因此不将其直接列为经过验证的复现命令。** 原始代码完整保留，详见 [归档说明](docs/ARCHIVE_NOTES.md)。

## 报告

[查看完整报告 PDF](paper/document2.pdf)。编译源文件可在安装中文 TeX 环境后执行：

```powershell
cd paper
xelatex document2.tex
xelatex document2.tex
```

## 验证范围

本次进行文件完整性校验、Python 语法检查及轻量计算/图表运行检查；具体结果见 `docs/VALIDATION.md`。**从现存敏感性 CSV 重算的部分最佳参数与已有汇总表不一致，两个版本未相互覆盖，差异已记录。** 已有 CSV 和图片来自原本地目录，不代表本次重新计算或完整复现了报告全部结论。随机仿真结果可能随环境与随机数状态变化。

本仓库未附加开源许可证；整理上传不改变原作者署名及权利。
