# 省级用电量预测（灰度关联 + 分布鲁棒优化）

本仓库实现了一个可复用的用电量预测流程，覆盖：

1. 明确预测粒度/预测期/评价指标  
2. 数据预处理（缺失、异常、标准化、时间对齐）  
3. 灰度关联分析（GRA）进行变量筛选  
4. 基准模型对比（仅历史序列、普通回归）  
5. 分布鲁棒优化（DRO）思想下的稳健线性模型  
6. 点预测 + 区间预测 + 情景预测  
7. 滚动回测与稳定性评估  
8. 新数据到来后的可重复训练与预测

## 安装

```bash
cd /path/to/project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据格式

输入 CSV 至少包含：

- `date`：日期列（可按月/日/年）
- `target`：用电量目标列
- 若干候选驱动变量列（如 `gdp`, `temperature`, `price` 等）

示例：

```csv
date,target,gdp,temperature,population,price,holiday,new_energy_capacity
2020-01-01,1100,5.6,2.1,7800,0.62,1,320
2020-02-01,1030,5.4,3.8,7802,0.62,1,325
```

## 运行

```bash
cd /path/to/project
python src/power_forecasting_pipeline.py \
  --data /absolute/path/to/electricity.csv \
  --date-col date \
  --target-col target \
  --horizon 6 \
  --top-k 5 \
  --robust-radius 1.0 \
  --scenario-delta 0.02 \
  --gra-rho 0.5 \
  --iqr-multiplier 1.5 \
  --interval-alpha 0.2 \
  --robust-candidates 0.01,0.1,1,3,10 \
  --output /absolute/path/to/output
```

输出文件：

- `metrics.json`：回测指标与模型对比
- `metrics.json` 中包含 `model_equations` 字段，可查看线性回归与 DRO 模型预测方程
- `grey_relation_scores.csv`：灰度关联度结果
- `future_forecast.csv`：未来点预测/区间预测/情景预测

## 方法说明（实现映射）

- 灰度关联：对候选变量计算与目标序列关联度，选择前 K 个变量
- DRO 模型：用稳健半径映射到正则化强度（Wasserstein-DRO 的常见近似形式）
- 区间预测：基于残差分位数构造预测区间
- 情景预测：通过特征扰动构造低/中/高负荷情景

## 备注

- 当前实现聚焦通用落地流程，便于在真实省级数据上直接迭代。
- 如需接入 ARIMA/LSTM/XGBoost，可在同一回测框架中扩展。

## Excel 售电量预测（长期趋势 + 季节系数 + 不规则系数）

针对 Excel（如仓库中的 `2020-2024日度、月度表.xlsx`），可使用 TSI（Trend-Seasonal-Irregular）分解预测：

```bash
cd /path/to/project
python src/sales_tsi_forecast.py \
  --data /absolute/path/to/2020-2024日度、月度表.xlsx \
  --sheet 2020-2024日度 \
  --date-col "日期" \
  --horizon 30 \
  --output /absolute/path/to/output
```

说明: 若 Excel 第一列无列名, pandas 可能自动命名为 `"Unnamed: 0"`, 此时可显式传入该列名; 更推荐在数据源中明确日期列名(如 `"日期"`)。

输出：

- 若识别到五类字段（`大工业`、`居民生活`、`农业生产`、`工商业`、`趸售及其他`）：
  - `大工业_result.csv`、`居民生活_result.csv`、`农业生产_result.csv`、`工商业_result.csv`、`趸售及其他_result.csv`：
    每类历史售电量 vs 拟合值对比 + 未来预测（均包含趋势/季节/不规则分量）
  - `summary_result.csv`：五类汇总表（历史对比 + 未来预测）
  - `metrics.json`：各类别误差指标与模型预测方程（`category_metrics`）
- 否则（单目标模式）：
  - `historical_vs_fitted.csv`：历史售电量与拟合值对比（含趋势/季节/不规则分量）
  - `future_sales_forecast.csv`：未来预测值（含趋势/季节/不规则分量）
  - `metrics.json`：历史拟合误差（MAE/RMSE/MAPE）与模型预测方程（`model_equations`）

补充说明：

- 若某列在尾部连续为空（如有效值到 2024/7，2024/8-2024/12 为空），脚本会将尾部空值日期识别为未来预测区间；
- 模型仅使用最后一个真实值及之前的数据进行拟合（期间缺失仍会补齐），并输出“历史拟合对比 + 对尾部空日期的未来预测”。
