# -*- coding: utf-8 -*-
import pandas as pd

# 读取两个预测结果文件
dann_df = pd.read_excel('out/hotmap/dann_sub_sample_predictions.xlsx')
rf_df = pd.read_excel('out/hotmap/rf_sub_sample_predictions.xlsx')

# 先合并 DANN 与 RF 的预测结果
merged = dann_df[['特征文本', 'DANN预测标签', '正确标签']].merge(
    rf_df[['特征文本', 'RF预测标签']],
    on='特征文本',
    how='inner'
)

# 只保留 DANN 与 RF 预测不一致的样本
diff = merged[merged['DANN预测标签'] != merged['RF预测标签']]

# 从不一致样本中随机选择10条，且同一诊断在所有预测列中出现不超过2次
from collections import Counter

shuffled = diff.sample(frac=1, random_state=2025).reset_index(drop=True)
selected = []
diag_counter = Counter()

for _, row in shuffled.iterrows():
    # 统计该行涉及的所有诊断
    diags = [row['DANN预测标签'], row['RF预测标签'], row['正确标签']]
    # 检查加入后是否有诊断达到3次
    temp = diag_counter.copy()
    for d in diags:
        temp[d] += 1
    if any(v >= 3 for v in temp.values()):
        continue
    selected.append(row)
    diag_counter = temp
    if len(selected) == 10:
        break

merged = pd.DataFrame(selected)

# 整理输出列：特征文本、DANN预测、RF预测、正确预测
result = merged[['特征文本', 'DANN预测标签', 'RF预测标签', '正确标签']]
result.columns = ['特征文本', 'DANN预测', 'RF预测', '正确预测']

# 保存结果
result.to_excel('out/hotmap/samples.xlsx', index=False)
print(f'已保存 {len(result)} 条样本到 out/hotmap/samples.xlsx')
print(result.to_string())
