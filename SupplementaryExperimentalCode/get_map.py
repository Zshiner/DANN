import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
from collections import OrderedDict

# ================================================================
#  全局画图设置 — Q1 SCI 期刊级别
# ================================================================
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8.5,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

# ================================================================
#  读取 & 解析数据
# ================================================================
df = pd.read_excel("out/hotmap/features.xlsx")

diag_cols = df.columns[1:]                          # 5 个诊断列
diag_labels = [f"D{i+1}" for i in range(len(diag_cols))]
rank_labels = [f"Top-{i+1}" for i in range(5)]

def parse_cell(cell):
    m = re.match(r"(.+?)（([\d.]+)）", str(cell))
    if m:
        return m.group(1).strip(), float(m.group(2))
    return str(cell).strip(), 0.0

dann_rows = df[df["模型-排序"].str.startswith("dann")]
rf_rows   = df[df["模型-排序"].str.startswith("rf")]

def collect_features(rows):
    result = {}
    for col in diag_cols:
        pairs = []
        for _, row in rows.iterrows():
            pairs.append(parse_cell(row[col]))
        result[col] = pairs
    return result

dann_data = collect_features(dann_rows)
rf_data   = collect_features(rf_rows)

# ================================================================
#  构建 S 编号映射（全局唯一）
# ================================================================
all_features = OrderedDict()
for data in [dann_data, rf_data]:
    for col in diag_cols:
        for name, _ in data[col]:
            if name not in all_features:
                all_features[name] = f"S{len(all_features)+1}"

# ================================================================
#  构建 5×5 矩阵  (Rank × Diagnosis)  +  标注矩阵
# ================================================================
def build_compact(data):
    val_mat   = np.zeros((5, len(diag_cols)))
    label_mat = np.empty((5, len(diag_cols)), dtype=object)
    for j, col in enumerate(diag_cols):
        for i, (name, val) in enumerate(data[col]):
            val_mat[i, j]   = val
            label_mat[i, j] = all_features[name]
    return val_mat, label_mat

dann_val, dann_lbl = build_compact(dann_data)
rf_val,   rf_lbl   = build_compact(rf_data)

# ================================================================
#  绘图 — 宽而短的双子图
# ================================================================
fig, axes = plt.subplots(
    1, 2,
    figsize=(7.2, 2.8),
    gridspec_kw={"wspace": 0.30},
)

cmap = mpl.colormaps["YlOrRd"]

for ax, val_mat, lbl_mat, title in zip(
    axes,
    [dann_val, rf_val],
    [dann_lbl, rf_lbl],
    ["(a) DANN", "(b) RF"],
):
    im = ax.imshow(val_mat, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # 单元格内标注：S编号 + 数值
    for i in range(5):
        for j in range(len(diag_cols)):
            v = val_mat[i, j]
            s = lbl_mat[i, j]
            txt_color = "white" if v > 0.60 else "black"
            ax.text(
                j, i,
                f"{s}\n{v:.2f}",
                ha="center", va="center",
                fontsize=7, color=txt_color,
                fontweight="medium",
                linespacing=1.15,
            )

    ax.set_xticks(range(len(diag_cols)))
    ax.set_xticklabels(diag_labels)
    ax.set_yticks(range(5))
    ax.set_yticklabels(rank_labels)
    ax.set_xlabel("Diagnosis", labelpad=4)
    ax.set_ylabel("Feature importance rank", labelpad=4)
    ax.set_title(title, pad=6, fontweight="bold")

    # 细网格线
    ax.set_xticks(np.arange(-0.5, len(diag_cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=3)

# 共用色条
cbar = fig.colorbar(
    im, ax=axes, fraction=0.015, pad=0.03, shrink=0.85,
)
cbar.set_label("Normalized feature contribution", fontsize=9)
cbar.ax.tick_params(labelsize=8)
cbar.outline.set_linewidth(0.5)

plt.savefig("out/hotmap/map.png", dpi=600)
plt.close()

# ================================================================
#  打印映射表（论文图注用）
# ================================================================
print("===== Diagnosis mapping =====")
for lbl, col in zip(diag_labels, diag_cols):
    print(f"  {lbl}: {col}")

print("\n===== Symptom / Feature mapping =====")
for feat, lbl in all_features.items():
    print(f"  {lbl}: {feat}")

print(f"\nTotal unique symptoms: {len(all_features)}")
print("Done. Saved to out/hotmap/map.png")
