# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# 读取特征重要性文件
dann_df = pd.read_excel('out/hotmap/dann_feature_importance.xlsx', index_col=0)
rf_df = pd.read_excel('out/hotmap/rf_feature_importance.xlsx', index_col=0)

# 随机选择5列（证型）
rng = np.random.RandomState(2025)
selected_cols = rng.choice(dann_df.columns, size=5, replace=False)
print('随机选择的5列:', list(selected_cols))

# 对每一列从高到低排序，取前5行的行名和归一化后的值
dann_top5 = {}
rf_top5 = {}
for col in selected_cols:
    # DANN: 排序后对整列做0-1归一化，再取前5
    dann_sorted = dann_df[col].sort_values(ascending=False)
    col_min, col_max = dann_sorted.min(), dann_sorted.max()
    dann_normed = (dann_sorted - col_min) / (col_max - col_min) if col_max != col_min else dann_sorted * 0
    dann_top5[col] = [(idx, val) for idx, val in zip(dann_normed.head(5).index, dann_normed.head(5).values)]

    # RF: 同样处理
    rf_sorted = rf_df[col].sort_values(ascending=False)
    col_min, col_max = rf_sorted.min(), rf_sorted.max()
    rf_normed = (rf_sorted - col_min) / (col_max - col_min) if col_max != col_min else rf_sorted * 0
    rf_top5[col] = [(idx, val) for idx, val in zip(rf_normed.head(5).index, rf_normed.head(5).values)]

# 构建输出DataFrame：10行5列
# 行名: dann-top1 ~ dann-top5, rf-top1 ~ rf-top5
# 列名: 选择的5个证型
# 值: 特征名（归一化值）
rows = []
row_names = []
for i in range(5):
    row_names.append(f'dann-top{i+1}')
    rows.append({col: f'{dann_top5[col][i][0]}（{dann_top5[col][i][1]:.2f}）' for col in selected_cols})
for i in range(5):
    row_names.append(f'rf-top{i+1}')
    rows.append({col: f'{rf_top5[col][i][0]}（{rf_top5[col][i][1]:.2f}）' for col in selected_cols})

result = pd.DataFrame(rows, index=row_names)
result.index.name = '模型-排序'

# 保存
result.to_excel('out/hotmap/features.xlsx')
print(f'\n已保存到 out/hotmap/features.xlsx')
print(result.to_string())
