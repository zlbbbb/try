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
  --robust-candidates 0.01,0.1,1,3,10 \
  --output /absolute/path/to/output
```

输出文件：

- `metrics.json`：回测指标与模型对比
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
