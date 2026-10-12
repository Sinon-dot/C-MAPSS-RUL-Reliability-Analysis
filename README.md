# C-MAPSS 航空发动机剩余寿命预测与可靠性分析

**Aircraft Engine Remaining Useful Life Prediction and Reliability Analysis**

基于 NASA C-MAPSS FD001 航空发动机退化模拟数据集，开展**剩余寿命（RUL）预测、传感器故障注入、鲁棒性测试、异常检测、数据修复、预测性能恢复评价与不确定性分析**。项目不仅关注正常数据上的预测精度，也关注输入出现异常时预测性能如何变化，以及检测和修复措施是否有效。

> **项目定位**：基于公开模拟数据的机器学习与智能系统可靠性实践。故障由实验人工注入；结果不代表真实航空发动机的适航认证、安全认证或现场性能。

## 1. 项目功能与实验流程

| 模块 | 实现内容 |
| --- | --- |
| RUL 预测 | Random Forest、XGBoost、LSTM 训练与比较 |
| 故障注入及鲁棒性测试 | Gaussian Noise、Bias、Sensor Stuck、Missing Data |
| 异常检测 | 缺失值检测、Sensor 11 连续卡死检测，计算 Precision / Recall / F1 |
| 数据修复 | Median、History、Random Forest 传感器估计 |
| 性能恢复评价 | 比较 Clean / Fault / Repaired 条件下的 MAE、RMSE 与误差恢复率 |
| 重复实验 | 5 个随机种子下的 20% 缺失数据修复对比 |
| 不确定性分析 | 经验预测区间、模型集成预测分歧与误差关系 |

```text
NASA C-MAPSS FD001
        │
        ▼
数据预处理与特征筛选
        │
        ▼
RUL 预测（RF / XGBoost / LSTM）─────────► 预测不确定性分析
        │
        ▼
传感器故障注入（Noise / Bias / Stuck / Missing）
        │
        ▼
鲁棒性与预测误差分析
        │
        ▼
异常检测（Missing / Stuck）
        │
        ▼
数据修复（Median / History / RF）
        │
        ▼
修复后 RUL 预测与恢复率评价
        │
        ▼
多随机种子重复实验
```

## 2. 数据集与评价设置

使用 NASA **C-MAPSS FD001** 数据集，包含 100 台训练发动机的完整退化运行序列、100 台测试发动机的截断运行序列，以及各测试发动机最后观测周期对应的真实 RUL。

原始数据**未包含在本仓库**，请自行取得并放在 `data/FD001/`：

```text
data/FD001/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

实验经过低方差筛选使用 14 个传感器特征；训练和验证按发动机编号划分，避免同一发动机的不同周期同时进入两个子集。LSTM 使用长度为 30 的历史滑动窗口。预测指标为 **MAE、RMSE**；异常检测指标为 **Precision、Recall、F1**。

> **模型比较口径**：LSTM 的滑动窗口会改变逐周期样本范围，因此其普通验证结果不能与树模型的全周期验证结果直接对比。下文三模型正式测试结果统一针对 FD001 官方测试集的 **100 台发动机最后观测周期**，但树模型使用当前周期特征、LSTM 使用历史时序输入。

## 3. RUL 预测模型比较

| 模型 | 测试 MAE ↓ | 测试 RMSE ↓ |
| --- | ---: | ---: |
| Random Forest | 25.108 | 34.376 |
| XGBoost | 23.536 | 32.941 |
| **LSTM** | **18.395** | **27.613** |

数值来源：[`results/test_model_results.csv`](results/test_model_results.csv)。本次测试中 LSTM 误差最低，但其输入形式不同，不应将结果直接解读为架构的普遍优越性。

![RUL 预测结果](results/RUL_prediction.png)

[MAE 对比图](results/model_comparison.png) · [RMSE 对比图](results/model_rmse_comparison.png)

> 部分早期可视化可能使用验证集口径；正式测试数值以 `test_model_results.csv` 为准。

## 4. 传感器故障注入与鲁棒性分析

| 故障 | 模拟方法 |
| --- | --- |
| Gaussian Noise | 在传感器读数上叠加高斯噪声 |
| Bias | 为传感器读数施加固定偏移 |
| Sensor Stuck | 将指定传感器读数冻结 |
| Missing Data | 随机将部分传感器数据设为缺失 |

XGBoost 原始鲁棒性实验：

| 条件 | MAE ↓ | RMSE ↓ |
| --- | ---: | ---: |
| Clean | 23.536 | 32.941 |
| Gaussian Noise (1.0) | 33.350 | 43.055 |
| Bias (1.0) | 22.381 | 28.834 |
| Sensor 11 Stuck | 28.526 | 35.448 |
| Missing 20% | 30.275 | 40.551 |

来源：[`results/robustness_summary.csv`](results/robustness_summary.csv)。偏移实验的一次误差下降不意味着偏移通常有利，其结果可能受偏移方向、强度及测试样本影响。

![传感器故障对预测误差的影响](results/fault_impact.png)

> **注意**：原始鲁棒性实验与第 5 节的检测/修复实验采用不同的故障注入范围与评价流程，不能跨实验直接计算恢复率。

## 5. 异常检测、数据修复和性能恢复

### 5.1 异常检测

在人工注入的 20% 随机缺失和 Sensor 11 连续卡死 20 周期的场景下，得到：

| 故障条件 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Missing 20% | 1.000 | 1.000 | 1.000 |
| Sensor 11 Stuck | 1.000 | 0.851 | 0.920 |

来源：[`results/fault_detection_metrics.csv`](results/fault_detection_metrics.csv)。缺失检测识别了 **36,668** 个注入缺失位置，主要因为 `NaN` 标识可直接观测，不能视为复杂异常识别能力的证明。卡死检测识别 **1,702 / 2,000** 个异常周期，100 个模拟卡死事件均被发现，平均检测延迟约 **2.98 个周期**。

![卡死异常检测](results/sensor_stuck_detection.png)

### 5.2 数据修复策略

- **Median**：使用训练集对应传感器的中位数填补缺失值。
- **History**：优先使用同一发动机历史有效读数前向填补；无可用历史时回退到训练集统计值。
- **RF Repair**：使用其他 13 个传感器预测 Sensor 11 的读数，对检测到的卡死位置进行修复。

修复效果用**同一个已训练的 XGBoost RUL 模型**评价，以减少预测器变化带来的混杂。

### 5.3 修复前后对比

| 实验条件 | MAE ↓ | RMSE ↓ |
| --- | ---: | ---: |
| Clean | 23.536 | 32.941 |
| Missing Fault 20% | 27.475 | 35.709 |
| Missing + Median Repair | 29.459 | 36.822 |
| Missing + History Repair | 23.672 | 32.749 |
| Sensor 11 Stuck (20 Cycles) | 27.511 | 36.457 |
| Stuck + RF Repair | 24.176 | 33.399 |

来源：[`results/repair_performance_comparison.csv`](results/repair_performance_comparison.csv)。

当故障使误差增加时，误差恢复率定义为：

$$
\mathrm{Recovery\ Rate}=\frac{E_{\mathrm{fault}}-E_{\mathrm{repaired}}}{E_{\mathrm{fault}}-E_{\mathrm{clean}}}\times100\%.
$$

其中 $E$ 为 MAE 或 RMSE。以 MAE 计算：**History 修复 Missing 的恢复率为 96.56%**，**RF 修复 Sensor 11 Stuck 为 83.89%**，而 **Median 修复 Missing 为 -50.36%**（修复后误差进一步增加）。此指标描述本组测试误差的变化，并非真实系统可靠性保证。

![修复前后 MAE](results/repair_mae_comparison.png)

![修复前后 RMSE](results/repair_rmse_comparison.png)

详细恢复率：[`results/repair_recovery_summary.csv`](results/repair_recovery_summary.csv)。

## 6. 多随机种子重复实验

固定 Missing 20% 故障比例、预测模型及修复策略，在随机种子 **42、43、44、45、46** 下重复实验：

| 条件 | MAE 均值 | MAE 标准差 | RMSE 均值 | RMSE 标准差 |
| --- | ---: | ---: | ---: | ---: |
| Clean | 23.536 | 0.000 | 32.941 | 0.000 |
| Fault | 28.051 | 0.878 | 36.078 | 1.458 |
| Median | 28.077 | 1.096 | 35.361 | 1.489 |
| **History** | **23.888** | **0.722** | **33.127** | **0.838** |

History 在 **5/5** 次实验中降低 MAE，五次 MAE 恢复率的算术平均约 **92.09%**；Median 的相应平均恢复率约 **-3.90%**。这些结果只反映当前数据集及特定缺失设置。

![多随机种子 MAE](results/missing_multiseed_mae.png)

![多随机种子 RMSE](results/missing_multiseed_rmse.png)

[汇总数据](results/missing_multiseed_summary.csv) · [逐次结果](results/missing_multiseed_results.csv) · [恢复率](results/missing_multiseed_recovery.csv)

## 7. 预测不确定性分析

- **经验预测区间**：根据验证残差构建目标覆盖水平为 90% 的经验区间，再检查测试样本中的实际覆盖表现；不声称具有严格的分布无关覆盖保证。
- **集成预测分歧**：以多个模型预测的标准差作为分歧程度，考察其与绝对预测误差的关系及故障条件下的变化。

![不同故障下的不确定性](results/uncertainty_fault.png)

![预测误差与集成分歧](results/uncertainty_error.png)

[预测区间明细](results/prediction_intervals_90.csv) · [集成分歧与误差数据](results/ensemble_uncertainty_error.csv)

## 8. 仓库结构

以下为当前 GitHub 仓库的主要文件。原始 FD001 数据需要使用者自行放入对应路径。

```text
C-MAPSS-RUL-Reliability-Analysis/
├── .gitattributes                    # 两个大模型的 Git LFS 跟踪规则
├── README.md
├── requirements.txt
├── data/
│   └── README.md                     # 原始数据获取及目录说明
├── notebooks/
│   └── 01_RUL_prediction_and_reliability_analysis.ipynb
├── models/
│   ├── lstm_model.pth
│   ├── random_forest_model.pkl       # Git LFS
│   ├── sensor11_repair_model.pkl     # Git LFS
│   └── xgboost_model.pkl
└── results/
    ├── test_model_results.csv
    ├── robustness_summary.csv
    ├── fault_detection_metrics.csv
    ├── repair_performance_comparison.csv
    ├── repair_recovery_summary.csv
    ├── missing_multiseed_results.csv
    ├── missing_multiseed_summary.csv
    ├── prediction_intervals_90.csv
    ├── ensemble_uncertainty_error.csv
    ├── RUL_prediction.png
    ├── fault_impact.png
    ├── sensor_stuck_detection.png
    ├── repair_mae_comparison.png
    ├── repair_rmse_comparison.png
    ├── missing_multiseed_mae.png
    ├── missing_multiseed_rmse.png
    ├── uncertainty_fault.png
    ├── uncertainty_error.png
    └── ...                           # 其余 CSV / PNG / 事件明细
```

### 模型文件与 Git LFS

**目前四个已训练模型均已上传至仓库。** 其中：

| 文件 | 功能 | 存储方式 |
| --- | --- | --- |
| `models/lstm_model.pth` | LSTM RUL 预测模型 | 普通 Git |
| `models/xgboost_model.pkl` | XGBoost RUL 预测模型 | 普通 Git |
| `models/random_forest_model.pkl` | 随机森林 RUL 预测模型 | **Git LFS** |
| `models/sensor11_repair_model.pkl` | Sensor 11 随机森林修复模型 | **Git LFS** |

完整获取大文件时，建议先安装 [Git](https://git-scm.com/) 和 [Git LFS](https://git-lfs.com/)，然后运行：

```bash
git lfs install
git clone https://github.com/Sinon-dot/C-MAPSS-RUL-Reliability-Analysis.git
cd C-MAPSS-RUL-Reliability-Analysis
git lfs pull
```

`git lfs pull` 用于确保下载到的是完整模型，而不是 Git LFS 指针文本。如果使用 GitHub 网站上的单个文件下载方式，请特别留意大文件是否完整。

> 已上传模型不等于在任意 Python 环境中都可以直接加载；pickle 模型可能受到 scikit-learn 等依赖版本的影响，复现时应尽量使用与训练时一致的环境。只加载可信来源的 `.pkl` 文件。

## 9. 安装与运行

### 9.1 准备环境

建议使用独立 Python 环境（例如 Anaconda）。在项目根目录运行：

```bash
python -m pip install -r requirements.txt
python -m pip install notebook
```

项目主要依赖 NumPy、Pandas、scikit-learn、XGBoost、PyTorch、Matplotlib 和 joblib。当前 `requirements.txt` 尚未锁定精确版本，可能需要根据本地训练环境调整依赖。

### 9.2 准备数据

从 NASA C-MAPSS 数据源获取 FD001 数据，放在**本地项目根目录**下的 `data/FD001/`，文件名如下：

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

### 9.3 运行实验

启动 Jupyter Notebook：

```bash
jupyter notebook
```

打开：

[`notebooks/01_RUL_prediction_and_reliability_analysis.ipynb`](notebooks/01_RUL_prediction_and_reliability_analysis.ipynb)

按 Notebook 的实际单元格顺序运行：**数据准备 → RUL 预测 → 故障注入与鲁棒性分析 → 异常检测 → 数据修复与恢复率 → 多种子实验 → 不确定性分析**。

运行前请检查 `PROJECT_ROOT`、`DATA_DIR`、`RESULTS_DIR`、`MODELS_DIR` 是否指向当前克隆仓库。Notebook 可能重新训练并覆盖 `models/` 中的模型文件，也可能覆盖 `results/` 中的同名 CSV/PNG，建议先备份已有输出。

> 本项目保留训练 Notebook 和已训练模型，但尚未验证所有依赖版本组合下的全流程从零复现。实验结果应以对应 CSV 及具体运行日志为准。

## 10. 局限性与后续工作

1. **场景范围**：当前仅验证 FD001 模拟数据，尚未进行真实机组故障和跨工况数据验证。
2. **特征选择**：低方差筛选先于训练/验证划分，虽不使用验证标签，但更严格的评估应将特征选择完全限制在训练折中。
3. **比较公平性**：三模型正式测试对象相同，但输入特征的时间范围不同；逐周期验证口径需特别区分。
4. **异常泛化能力**：Missing 的 `NaN` 可直接识别，Stuck 检测主要针对人工构造的规则性故障。
5. **统计稳定性**：当前仅对 Missing 20% 进行 5 个种子的重复实验，还需扩展故障强度和场景。
6. **不确定性可靠性**：经验区间没有严格覆盖保证；需要更多校准和覆盖率评估。
7. **工程复现**：尚待固定依赖版本、拆分 Notebook 功能模块并增加自动化测试。

## 11. 总结

本项目围绕航空发动机 RUL 预测，形成了**预测建模—故障注入—异常检测—数据修复—预测恢复评价—不确定性分析**的完整实验链条。结果表明，在本组模拟故障实验中，历史值填补和随机森林传感器估计能够改善下游 RUL 预测误差，而中位数填补不一定有效。项目重点是理解模型在异常输入下的表现及修复效果，为机器学习系统可靠性测试提供可复用的实验案例。
