import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from collections import Counter

sns.set_theme(style="ticks", context="notebook", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['mathtext.default'] = 'regular'

c_shifts = []
h_shifts = []
solvents_c = []
solvents_h = []
multiplicities = []  

# 载入数据
with open("./dataset/NMRexp_spectra_dataset.pkl", "rb") as f:
    spectra_list = pickle.load(f)


# 分析数据
for spectra in spectra_list:
    for peak in spectra['13C_NMR']['peaks']:
        try:
            c_shifts.append(float(peak['shift']))
        except (ValueError, TypeError):
            pass
    solvents_c.append(spectra['13C_NMR']['solvent'])

    for peak in spectra['1H_NMR']['peaks']:
        try:
            h_shifts.append(float(peak['shift']))
        except (ValueError, TypeError):
            pass
        m = peak.get('multiplicity')
        if m and isinstance(m, str) and m.strip():
            multiplicities.append(m.strip().lower())

    solvents_h.append(spectra['1H_NMR']['solvent'])


# ── 构建 DataFrame ───────────────────────────────────────────────
all_solvents = solvents_c + solvents_h
solvent_counts = pd.Series(all_solvents).value_counts().head(10).reset_index()
solvent_counts.columns = ['Solvent', 'Count']

# multiplicity 统计，取 Top 15
mult_counts = pd.DataFrame(
    Counter(multiplicities).most_common(15),
    columns=['Multiplicity', 'Count']
)

# ── 绘图 ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 16))  # ← 改为 2×2 布局
axes = axes.flatten()

# 1. 13C 化学位移分布
sns.histplot(
    c_shifts,
    bins=100,
    binrange=(-20, 220),
    color='#1f77b4',
    edgecolor='white',
    alpha=0.7,
    ax=axes[0]
)
axes[0].set_title('13C NMR Chemical Shift Distribution', fontsize=16)
axes[0].set_xlabel('Chemical Shift (ppm)', fontsize=14)
axes[0].set_ylabel('Number of Peaks', fontsize=14)
axes[0].set_xlim(-20, 220)

# 2. 1H 化学位移分布
sns.histplot(
    h_shifts,
    bins=100,
    binrange=(-2, 12),
    color='#ff7f0e',
    edgecolor='white',
    alpha=0.7,
    ax=axes[1]
)
axes[1].set_title('1H NMR Chemical Shift Distribution', fontsize=16)
axes[1].set_xlabel('Chemical Shift (ppm)', fontsize=14)
axes[1].set_ylabel('Number of Peaks', fontsize=14)
axes[1].set_xlim(-2, 12)

# 3. 溶剂分布 Top10
sns.barplot(
    data=solvent_counts,
    y='Solvent',
    x='Count',
    color='#2ca02c',
    alpha=0.7,
    orient='h',
    ax=axes[2]
)
axes[2].set_title('Top 10 Solvent Distribution', fontsize=16)
axes[2].set_xlabel('Number of Samples', fontsize=14)
axes[2].set_ylabel('')

# 4. 1H 的耦合裂分种类
sns.barplot(
    data=mult_counts,
    y='Multiplicity',
    x='Count',
    color='#9467bd',
    alpha=0.7,
    orient='h',
    ax=axes[3]
)
axes[3].set_title('1H NMR Multiplicity Distribution (Top 15)', fontsize=16)
axes[3].set_xlabel('Number of Peaks', fontsize=14)
axes[3].set_ylabel('')

for i, (count, mult) in enumerate(zip(mult_counts['Count'], mult_counts['Multiplicity'])):
    axes[3].text(
        count + mult_counts['Count'].max() * 0.01,
        i,
        f'{count:,}',
        va='center',
        fontsize=10
    )

plt.tight_layout()
plt.savefig('img/spectra_distribution.png', dpi=300, bbox_inches='tight')
plt.show()


# ── 统计信息 ─────────────────────────────────────────────────────
unique_solvents = pd.Series(all_solvents).value_counts()
print(f"Total 13C peaks:      {len(c_shifts):>10,}")
print(f"Total 1H peaks:       {len(h_shifts):>10,}")
print(f"Unique solvents:      {len(unique_solvents):>10,}")
print(f"Most common solvent:  {unique_solvents.idxmax()} ({unique_solvents.max():,} samples)")

# ── 所有 Multiplicity 种类 ────────────────────────────────────────
all_mult = pd.Series(multiplicities).value_counts()

print(f"\nTotal unique multiplicities: {len(all_mult)}")
print("\nAll multiplicities (sorted by frequency):")
for mult, count in all_mult.items():
    print(f"  {mult:<15} {count:>10,}")
    
"""
All multiplicities (sorted by frequency):
  m                3,771,784
  d                2,413,349
  s                2,284,015
  t                  917,450
  dd                 883,264
  ddd                152,458
  q                  152,173
  dt                 121,295
  td                 112,522
  brs                110,238
  br                  33,775
  dq                  28,002
  p                   23,035
  ddt                 22,779
  tt                  20,997
  hept                14,973
  qd                  13,449
  dddd                12,711
  dtd                  7,442
  brd                  5,614
  tdd                  4,932
  app t                3,836
  hex                  2,784
  qt                   2,483
  ddq                  2,346
  dtt                  2,005
  app d                1,999
  tq                   1,965
  dp                   1,891
  dqd                  1,561
  brt                  1,497
  abq                  1,289
  ab                   1,150
  pd                   1,056
  app q                  911
  app dt                 855
  app td                 813
  app s                  740
  dddt                   725
  abd                    688
  qq                     615
  qdd                    583
  app dd                 495
  tdt                    478
  brdd                   336
  dh                     316
  ttd                    316
  pt                     290
  ddddd                  279
  ddtd                   245
  app dq                 229
  heptd                  220
  brq                    218
  ddp                    177
  dtdd                   176
  spt                    160
  app p                  158
  dtq                    156
  tp                     149
  dqt                    131
  app ddt                116
  abx                    115
  tddd                   114
  app qd                  87
  nonet                   86
  oct                     73
  aa'bb'                  71
  hd                      67
  qdt                     65
  tdq                     64
  dddq                    61
  qqd                     60
  app dtd                 57
  dqq                     56
  dpd                     34
  br dt                   32
  ddtt                    30
  app tdd                 29
  br td                   18
  adt                      7
"""
