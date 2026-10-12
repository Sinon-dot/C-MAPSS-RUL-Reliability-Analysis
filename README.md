# C-MAPSS 航空发动机剩余寿命预测与可靠性分析

**Aircraft Engine Remaining Useful Life Prediction and Reliability Analysis**

基于 NASA C-MAPSS FD001 数据集，构建包含**剩余寿命（RUL）预测、传感器故障注入、异常检测、数据修复、预测性能恢复评价和不确定性分析**的实验流程。项目重点关注：传感器异常如何影响预测模型，以及检测、修复措施能否改善下游 RUL 预测的可靠性。

> 本项目为基于公开模拟数据的研究与工程实践，实验故障为人工注入，结果不代表真实航空发动机系统的安全认证结论。

## 1. 项目概览

| 模块 | 主要内容 |
| --- | --- |
| RUL 预测 | Random Forest、XGBoost、LSTM 三模型训练与比较 |
| 鲁棒性测试 | Gaussian Noise、Bias、Sensor Stuck、Missing Data 故障注入 |
| 异常检测 | 缺失值检测、传感器连续不变值检测，统计 Precision / Recall / F1 |
| 异常修复 | 训练集统计量填补、历史值填补、随机森林传感器估计 |
| 性能恢复评价 | 比较 Clean / Fault / Repaired 条件下的 MAE、RMSE 和恢复率 |
| 重复实验 | 5 组随机种子下的 20% 缺失数据修复实验 |
| 不确定性分析 | 经验预测区间、集成预测分歧与预测误差的关系 |

### 实验流程

```text
NASA C-MAPSS FD001
        |
        v
数据预处理与特征筛选
        |
        v
RUL 预测模型训练与评价
(Random Forest / XGBoost / LSTM)
        |
        +----------------------+
        |                      |
        v                      v
传感器故障注入             预测不确定性分析
(Noise/Bias/Stuck/Missing)  (区间/集成分歧)
        |
        v
鲁棒性测试与性能退化分析
        |
        v
异常检测（Missing / Stuck）
        |
        v
数据修复（Median / History / RF）
        |
        v
修复后 RUL 预测与恢复率评价
        |
        v
5 组随机种子重复实验
```

## 2. 数据集与实验设置

使用 **NASA C-MAPSS FD001** 航空发动机退化模拟数据集。**当前 GitHub 仓库未包含原始数据文件，运行前需要自行获取并放入本地 `data/FD001/`。** 数据由以下三部分组成：

- `train_FD001.txt`：100 台发动机的完整退化运行序列，用于模型训练与验证。
- `test_FD001.txt`：100 台发动机的截断运行序列，用于测试。
- `RUL_FD001.txt`：对应 100 台测试发动机最后一个观测周期的真实 RUL。

在 FD001 数据上进行低方差特征筛选后，使用 14 个传感器特征。训练与验证按发动机编号划分，避免同一发动机的周期样本同时进入训练和验证子集；LSTM 采用长度为 30 的历史滑动窗口。

**评价指标：** 平均绝对误差（MAE）、均方根误差（RMSE）；故障检测另外报告 Precision、Recall、F1。

> **比较口径说明：** LSTM 因滑动窗口而无法在每台发动机最初的 29 个周期形成完整输入，因此普通逐周期验证结果与随机森林/XGBoost的全周期验证结果不能直接比较。仓库保留了对齐预测目标的 `validation_model_comparison.csv`；下文优先展示三模型在**同一批 100 台测试发动机**上的正式测试结果。

## 3. RUL 预测模型与结果

构建三种模型：

- **Random Forest（随机森林）**：基于当前周期的传感器特征进行回归预测。
- **XGBoost**：基于当前周期的传感器特征进行梯度提升回归预测。
- **LSTM**：基于连续 30 个周期的传感器序列学习退化时序信息。

### 正式测试集表现

以下数值来自 [`results/test_model_results.csv`](results/test_model_results.csv)，评价目标均为 **FD001 官方测试集 100 台发动机最后观测周期的 RUL**。

| 模型 | MAE ↓ | RMSE ↓ |
| --- | ---: | ---: |
| Random Forest | 25.108 | 34.376 |
| XGBoost | 23.536 | 32.941 |
| **LSTM** | **18.395** | **27.613** |

在本次实验中，LSTM 的 MAE、RMSE 最低。三模型使用相同的测试对象与标签，但输入形式不同：树模型使用当前周期特征，LSTM 使用历史序列。

### 预测结果可视化

![RUL 预测结果](results/RUL_prediction.png)

[模型 MAE 对比图](results/model_comparison.png) · [模型 RMSE 对比图](results/model_rmse_comparison.png)

> 仓库中的部分早期可视化可能采用验证集绘图口径；三模型正式测试数值请以本节表格和 `test_model_results.csv` 为准。

## 4. 传感器故障注入与鲁棒性分析

为研究输入异常对预测模型的影响，在传感器数据中模拟以下故障：

| 故障类型 | 模拟方式 |
| --- | --- |
| Gaussian Noise | 对传感器读数添加高斯噪声 |
| Bias | 对传感器读数施加固定偏移 |
| Sensor Stuck | 将指定传感器读数冻结 |
| Missing Data | 随机缺失部分传感器读数 |

以下为 XGBoost 在原始鲁棒性测试中的结果，来自 [`results/robustness_summary.csv`](results/robustness_summary.csv)：

| 条件 | MAE ↓ | RMSE ↓ |
| --- | ---: | ---: |
| Clean | 23.536 | 32.941 |
| Gaussian Noise (1.0) | 33.350 | 43.055 |
| Bias (1.0) | 22.381 | 28.834 |
| Sensor 11 Stuck | 28.526 | 35.448 |
| Missing 20% | 30.275 | 40.551 |

不同异常对模型影响不同；单次偏移实验中误差下降并不意味着该异常通常有益，而可能与故障方向、强度和有限样本有关。

![传感器故障对预测的影响](results/fault_impact.png)

> **注意：** 本节是原始鲁棒性实验；第 5 节的异常检测与修复实验在完整测试序列上构造故障，再提取每台发动机的末周期输入进行 RUL 评价。两组实验的注入范围/方式不同，因此即使故障名称相同，MAE、RMSE 也不要求一致，不能跨实验直接计算恢复率。

## 5. 异常检测与修复

### 5.1 异常检测

针对 20% 随机数据缺失，利用 `NaN` 标识检测缺失位置；针对 **Sensor 11 连续卡死 20 个周期**，利用连续相同读数的规则检测异常。

检测结果来自 [`results/fault_detection_metrics.csv`](results/fault_detection_metrics.csv)：

| 故障类型 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Missing 20% | 1.000 | 1.000 | 1.000 |
| Sensor 11 Stuck | 1.000 | 0.851 | 0.920 |

在本次人工注入的测试序列中：

- 缺失值检测识别了全部 **36,668** 个注入缺失位置；这一结果主要反映 `NaN` 可直接观测，并非复杂异常识别能力。
- 卡死检测共识别 **1,702 / 2,000** 个异常周期；100 个模拟卡死事件均被检测到，平均检测延迟约 **2.98 个周期**。
- 卡死检测依赖规则及阈值，仍需在不同故障长度、传感器类型和自然异常条件下验证。

![Sensor 11 卡死异常检测](results/sensor_stuck_detection.png)

检测明细见 [`stuck_detection_detail.csv`](results/stuck_detection_detail.csv) 和 [`stuck_detection_events.csv`](results/stuck_detection_events.csv)。

### 5.2 异常数据修复

在已有检测与故障输入基础上，比较三种修复方式：

1. **Median**：使用训练集对应传感器的中位数填补缺失值。
2. **History**：优先使用同一台发动机之前的有效读数前向填补；无历史值时回退到训练集统计值。
3. **Random Forest Repair**：使用其他 13 个传感器的读数估计 Sensor 11，并对检出的卡死位置进行修复。

修复后使用**同一个已训练的 XGBoost RUL 模型**评价，避免因重新训练预测模型而混淆修复效果。

### 5.3 修复前后 RUL 预测性能

结果来自 [`results/repair_performance_comparison.csv`](results/repair_performance_comparison.csv)：

| 实验条件 | MAE ↓ | RMSE ↓ |
| --- | ---: | ---: |
| Clean | 23.536 | 32.941 |
| Missing Fault 20% | 27.475 | 35.709 |
| Missing + Median Repair | 29.459 | 36.822 |
| Missing + History Repair | 23.672 | 32.749 |
| Sensor 11 Stuck (20 Cycles) | 27.511 | 36.457 |
| Stuck + RF Repair | 24.176 | 33.399 |

定义误差恢复率为：

$$
\text{Recovery Rate} = \frac{E_{\text{fault}}-E_{\text{repaired}}}{E_{\text{fault}}-E_{\text{clean}}}\times 100\%
$$

其中 $E$ 可取 MAE 或 RMSE，且该指标仅在故障确实使误差增大时具有这里的恢复意义。

**本次实验的 MAE 恢复结果：**

- **History 修复 Missing：96.56%**；
- **RF 修复 Sensor 11 Stuck：83.89%**；
- **Median 修复 Missing：-50.36%**，即修复后反而进一步增加误差。

恢复率超过 100% 仅表示当前测试样本上修复后的误差低于 Clean 基线，不代表真实数据质量优于无故障数据。

![修复前后 MAE 对比](results/repair_mae_comparison.png)

![修复前后 RMSE 对比](results/repair_rmse_comparison.png)

完整恢复率见 [`results/repair_recovery_summary.csv`](results/repair_recovery_summary.csv)。

## 6. 多随机种子重复实验

为检查随机缺失位置对修复结论的影响，固定模型和故障比例（Missing 20%），在 **5 个随机种子（42、43、44、45、46）** 下重复比较 Clean、Fault、Median 和 History 四种条件。

结果来自 [`results/missing_multiseed_summary.csv`](results/missing_multiseed_summary.csv)：

| 方法 | MAE 均值 | MAE 标准差 | RMSE 均值 | RMSE 标准差 |
| --- | ---: | ---: | ---: | ---: |
| Clean | 23.536 | 0.000 | 32.941 | 0.000 |
| Fault | 28.051 | 0.878 | 36.078 | 1.458 |
| Median | 28.077 | 1.096 | 35.361 | 1.489 |
| **History** | **23.888** | **0.722** | **33.127** | **0.838** |

- History 方法在 **5/5** 组实验中均降低了 MAE，五组恢复率的算术平均值约 **92.09%**。
- Median 方法在不同种子下表现不稳定，五组 MAE 恢复率的算术平均值约 **-3.90%**。
- 以上结果仅反映当前 FD001、缺失比例和模型设置下的重复实验，不能替代跨数据集验证。

![多随机种子 MAE 对比](results/missing_multiseed_mae.png)

![多随机种子 RMSE 对比](results/missing_multiseed_rmse.png)

原始逐次结果及恢复率：[实验明细](results/missing_multiseed_results.csv) · [恢复率明细](results/missing_multiseed_recovery.csv)。

## 7. 预测不确定性分析

除了平均预测误差，项目还从两个角度分析预测的不确定性：

- **经验预测区间**：根据验证残差构建 90% 目标覆盖水平的经验误差区间，并在测试集上检查覆盖情况；此处不声称具有严格的分布无关覆盖保证。
- **集成预测分歧**：考察多个模型预测的标准差及其与绝对预测误差的关系，并对比故障场景下的不确定性变化。

相关输出：

- [`prediction_intervals_90.csv`](results/prediction_intervals_90.csv)：预测区间明细。
- [`ensemble_uncertainty_error.csv`](results/ensemble_uncertainty_error.csv)：真实 RUL、预测 RUL、绝对误差与集成不确定性。

![不同故障条件下的不确定性](results/uncertainty_fault.png)

![预测误差与不确定性的关系](results/uncertainty_error.png)

## 8. 项目结构

```text
C-MAPSS-RUL-Reliability-Analysis/
├── data/                         # 需在本地准备；当前仓库未上传
│   └── FD001/
│       ├── train_FD001.txt
│       ├── test_FD001.txt
│       └── RUL_FD001.txt
├── notebooks/
│   └── 01_RUL_prediction_and_reliability_analysis.ipynb
├── models/
│   ├── lstm_model.pth
│   └── xgboost_model.pkl
├── results/
│   ├── test_model_results.csv
│   ├── validation_model_comparison.csv
│   ├── robustness_summary.csv
│   ├── fault_detection_metrics.csv
│   ├── stuck_detection_events.csv
│   ├── repair_performance_comparison.csv
│   ├── repair_recovery_summary.csv
│   ├── missing_multiseed_results.csv
│   ├── missing_multiseed_summary.csv
│   ├── missing_multiseed_recovery.csv
│   ├── prediction_intervals_90.csv
│   ├── ensemble_uncertainty_error.csv
│   ├── model_comparison.png
│   ├── model_rmse_comparison.png
│   ├── RUL_prediction.png
│   ├── fault_impact.png
│   ├── sensor_stuck_detection.png
│   ├── repair_mae_comparison.png
│   ├── repair_rmse_comparison.png
│   ├── missing_multiseed_mae.png
│   ├── missing_multiseed_rmse.png
│   ├── uncertainty_fault.png
│   ├── uncertainty_error.png
│   └── ...（其他实验 CSV 与明细）
├── requirements.txt
└── README.md
```

> **模型文件说明：** GitHub 当前仅保存较小的 `lstm_model.pth` 和 `xgboost_model.pkl`。随机森林 RUL 模型 `random_forest_model.pkl` 及传感器修复模型 `sensor11_repair_model.pkl` 因文件较大未上传；可通过 Notebook 的相应训练步骤在本地生成。请不要将“模型未上传”理解为实验未完成。

## 9. 环境与运行方法

### 9.1 安装依赖

建议在 Python 3 环境中运行（例如 Anaconda / Jupyter Notebook）。安装项目依赖：

```bash
python -m pip install -r requirements.txt
```

如尚未安装 Jupyter Notebook，可额外安装：

```bash
python -m pip install notebook
```

项目主要使用 NumPy、Pandas、scikit-learn、XGBoost、PyTorch、Matplotlib、joblib。

### 9.2 准备数据

请先自行获取 NASA C-MAPSS FD001 数据，并在本地项目根目录建立 `data/FD001/` 文件夹，确保其中包含：

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

### 9.3 运行 Notebook

打开：

```text
notebooks/01_RUL_prediction_and_reliability_analysis.ipynb
```

按 Notebook 的章节顺序执行实验：数据预处理 → 模型训练与评价 → 故障注入和鲁棒性分析 → 异常检测 → 数据修复 → 恢复率评价 → 重复实验与不确定性分析。

运行过程中会向 `results/` 保存 CSV 和 PNG；**同名文件会被覆盖**。需要保留已有实验数值时，请先备份结果文件夹。部分模型训练与实验可能需要一定运行时间，LSTM 重新训练的指标也可能随随机状态变化。

## 10. 局限性与后续工作

1. **数据与场景：** 当前实验仅使用 FD001 模拟数据，缺少真实发动机故障数据和跨工况验证。
2. **特征筛选：** 当前低方差筛选先于发动机级训练/验证划分执行，虽不使用验证标签，但仍存在验证分布参与特征选择的问题；严格研究应将筛选纳入训练折内部。
3. **模型公平比较：** 验证集逐周期样本与 LSTM 窗口样本数量不同，需使用对齐目标的结果；正式测试集采用相同 100 台发动机的末周期标签，但模型输入形式仍不同。
4. **异常检测：** Missing 的 `NaN` 检测较直接；Stuck 规则针对当前人工构造的故障形式，不能据此推断对复杂自然故障的泛化能力。
5. **修复稳定性：** 仅在 Missing 20% 场景下进行了 5 组随机种子实验；其他故障强度、传感器及修复方法仍需扩展。
6. **不确定性与复现：** 预测区间尚不具备严格的覆盖保证；LSTM 随机性和依赖版本固定等复现问题可进一步完善。

## 11. 总结

本项目将航空发动机 RUL 预测与传感器故障下的可靠性分析相结合，不仅比较不同预测模型的精度，也进一步验证了**“异常检测—输入修复—下游预测恢复”**这一流程。

在当前 FD001 实验中，历史值填补与随机森林传感器修复能够改善相应故障下的 RUL 预测表现，而简单中位数填补并不总是有效。项目同时通过重复实验和不确定性分析，展示了对模型性能、故障影响和修复效果的综合评价方法。
