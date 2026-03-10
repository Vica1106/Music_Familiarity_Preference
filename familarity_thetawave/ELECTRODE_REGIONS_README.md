# Music Familiarity & Theta EEG — 项目说明 / Project Summary

本文档汇总电极区域映射、分析流程、输出文件及运行方式。  
This document summarizes electrode region mapping, analysis pipeline, outputs, and how to run the scripts.

---

## 一、电极区域映射 / Electrode region mapping

### 中文

**数据来源**  
每个被试的 EEG 数据目录下有一份 `*_electrodes.tsv`，例如：  
`data/sub-020/ses-02/eeg/sub-020_ses-02_task-MusicListening_run-2_electrodes.tsv`  
内容为 EGI 128 通道帽的电极坐标，列：`name`, `x`, `y`, `z`。

**映射规则（按 z 坐标）**  
本数据集使用 **EGI 128 HCGSN** 电极帽，坐标中 **z 轴表示头部的“前–后/上”方向**（z 越大越靠前/上，越小越靠后）。区域划分在 `analysis_theta_familiarity.py` 的 `get_electrode_regions()` 中实现：

| 区域 region | 条件 condition | 含义 meaning |
|-------------|----------------|--------------|
| **frontal** | z ≥ 5 | 前部/额叶 |
| **central** | -2 ≤ z < 5 | 中部/中央区 |
| **parietal** | z < -2 | 后部/顶枕区 |

**E129** 为参考电极 (Cz)，不参与区域平均，在代码里被跳过。  
项目根目录下的 `electrode_regions.csv` 为按上述规则预生成的 E1–E128 映射表，仅作参考；**实际分析时区域在代码里按每份 electrodes.tsv 的 z 实时计算**，不依赖该 CSV。

### English

**Data source**  
Each run has an `*_electrodes.tsv` (e.g. under `data/sub-XXX/ses-YY/eeg/`) with EGI 128 channel positions: columns `name`, `x`, `y`, `z`.

**Mapping rule (by z)**  
**EGI 128 HCGSN** cap; **z** encodes anterior–posterior/superior (higher z = more anterior/superior). Regions in `get_electrode_regions()`:

- **frontal**: z ≥ 5  
- **central**: -2 ≤ z < 5  
- **parietal**: z < -2  

**E129** (Cz reference) is excluded from region averages. The file `electrode_regions.csv` in the project root is a pre-generated reference; analysis computes regions on the fly from each run’s electrodes file.

---

## 二、如何运行 / How to run

### 中文

**依赖**  
```bash
pip install -r requirements.txt   # numpy, pandas, scipy, matplotlib, mne, tqdm
```

**主分析（familiarity vs theta）**  
```bash
# 小数据快速测试（仅前 N 个 trial）
python analysis_theta_familiarity.py --max-runs 5

# 全量分析
python analysis_theta_familiarity.py
```  
输出：`results/theta_by_familiarity.csv`、箱线图、脑地形图、统计结果与时间戳日志 `results/analysis_YYYYMMDD_HHMMSS.log`。

**探索性分析（事件分段 + TFR + 多图）**  
```bash
python explore_theta_familiarity.py
python explore_theta_familiarity.py --sub 1 --ses 2
python explore_theta_familiarity.py --max-runs 10   # 额外生成 theta vs 熟悉度组别箱线图
```

### English

**Dependencies**  
```bash
pip install -r requirements.txt   # numpy, pandas, scipy, matplotlib, mne, tqdm
```

**Main analysis (familiarity vs theta)**  
```bash
# Quick test on first N runs
python analysis_theta_familiarity.py --max-runs 5

# Full analysis
python analysis_theta_familiarity.py
```  
Outputs: `results/theta_by_familiarity.csv`, boxplots, topomaps, statistics, and log `results/analysis_YYYYMMDD_HHMMSS.log`.

**Exploratory analysis (event-based epochs, TFR, multiple plots)**  
```bash
python explore_theta_familiarity.py
python explore_theta_familiarity.py --sub 1 --ses 2
python explore_theta_familiarity.py --max-runs 10   # also boxplot theta vs familiarity group
```

---

## 三、主分析内容 / Main analysis pipeline

### 中文

**研究问题**  
音乐熟悉度（Familiarity）是否会影响 theta 频段（4–8 Hz）的 EEG 活动？比较 **Low 熟悉度（1–2 分）** 与 **High 熟悉度（5 分）** 的 theta 功率；可选分脑区（额/中央/顶）。

**数据预处理**  
- 行为：只保留 Familiarity = 1、2、5；3、4 剔除。分组：**Low** = 1,2，**High** = 5。  
- EEG：对每个保留试次加载对应 `.set` 与 `*_electrodes.tsv`，按 z 分额/中央/顶区，4–8 Hz 带通滤波后算每通道 theta 功率，再按脑区平均，得到 theta_frontal、theta_central、theta_parietal。  
- 汇总表：`results/theta_by_familiarity.csv`（Subject, Song_ID, Familiarity, Familiarity_Group, theta_frontal, theta_central, theta_parietal）。

**统计分析**  
对 Low vs High 在三个脑区上分别做：**独立样本 t 检验**、**Mann–Whitney U 检验**；输出 n、均值、标准差、统计量与 p 值。

**主分析输出图**  
1. **主结果图**：`theta_power_vs_familiarity_group.png` — 额区 theta 功率 × 熟悉度组别（Low/High）箱线图。  
2. **分脑区图**：`theta_power_by_region.png` — 额/中央/顶三列箱线图。  
3. **脑地形图**：`theta_topomap.png`（全 trial 平均 theta）；`theta_topomap_by_familiarity_group.png`（Low vs High 并排，两组均有数据时生成）。

### English

**Research question**  
Does music familiarity affect theta-band (4–8 Hz) EEG? Compare **Low familiarity (1–2)** vs **High familiarity (5)** theta power; optionally by region (frontal/central/parietal).

**Preprocessing**  
- Behaviour: keep only Familiarity 1, 2, 5; group **Low** (1,2), **High** (5).  
- EEG: load `.set` and electrodes per run, assign channels to frontal/central/parietal by z, bandpass 4–8 Hz, compute theta power per channel then average by region → theta_frontal, theta_central, theta_parietal.  
- Table: `results/theta_by_familiarity.csv`.

**Statistics**  
Per region: **t-test** and **Mann–Whitney U** (Low vs High); report n, mean, std, test stats, p.

**Main analysis figures**  
1. **Main**: `theta_power_vs_familiarity_group.png` — frontal theta vs familiarity group (boxplot).  
2. **By region**: `theta_power_by_region.png` — three boxplots (frontal/central/parietal).  
3. **Topography**: `theta_topomap.png` (average theta); `theta_topomap_by_familiarity_group.png` (Low vs High when both groups have data).

---

## 四、探索性分析输出 / Exploratory analysis outputs

### 中文

脚本 `explore_theta_familiarity.py` 采用与 enjoyment/alpha 类似的流程：基于 events.tsv 找音乐开始、预处理、分段、Morlet TFR、频段功率；重点为 **familiarity + theta**，并含 alpha 等对比。  
所有图保存在 `results/` 下：

| 文件名 | 说明 |
|--------|------|
| `explore_spectrogram_theta.png` | 单通道时–频图（theta 可见）。 |
| `explore_all_bands_over_time.png` | 五频段（delta/theta/alpha/beta/gamma）功率随时间，与“Frequency Band Power Over Time”同风格。 |
| `explore_theta_alpha_over_time.png` | 额区 theta 与 alpha 功率随时间。 |
| `explore_theta_by_region_over_time.png` | 额/中央/顶 theta 功率随时间。 |
| `explore_theta_vs_familiarity_group.png` | theta（额区）vs 熟悉度组别箱线图（需 `--max-runs > 0`）。 |
| `explore_topomap_familiarity_low_vs_high.png` | 类似 Graph 7：Low vs High 熟悉度各 3 个 example trial 的 theta 地形图（相对 scaling，RdBu_r）。 |

日志：`results/explore_YYYYMMDD_HHMMSS.log`。

### English

Script `explore_theta_familiarity.py` uses event-based epoching, preprocessing, Morlet TFR, and band power (same style as enjoyment/alpha); focus **familiarity + theta**, with alpha for comparison. All figures in `results/`:

| File | Description |
|------|-------------|
| `explore_spectrogram_theta.png` | Single-channel time–frequency (theta visible). |
| `explore_all_bands_over_time.png` | All five bands over time (“Frequency Band Power Over Time” style). |
| `explore_theta_alpha_over_time.png` | Frontal theta & alpha over time. |
| `explore_theta_by_region_over_time.png` | Theta over time by frontal/central/parietal. |
| `explore_theta_vs_familiarity_group.png` | Boxplot theta vs familiarity group (when `--max-runs > 0`). |
| `explore_topomap_familiarity_low_vs_high.png` | Graph 7–style: theta topomaps for 3 Low vs 3 High example trials (relative scaling, RdBu_r). |

Log: `results/explore_YYYYMMDD_HHMMSS.log`.

---

## 五、运行时的预期行为与常见提示 / Expected behaviour and common messages

### 中文

| 现象 | 是否预期 | 说明 |
|------|----------|------|
| Trials: 5 (Low: 5, High: 0) | ✅ 预期 | `--max-runs 5` 只取前 5 行，可能全是 Low；要看 High 需更多 runs 或全量。 |
| Omitted/Limited annotation(s) outside data range | ✅ 预期 | EEGLAB 事件有时超出数据长度，MNE 会忽略/截断；脚本已屏蔽此类警告。 |
| boundary events, data discontinuities | ✅ 预期 | 数据来自连续记录，含 boundary 正常；整段算 theta 不依赖事件。 |
| Error ... [Errno 60] Operation timed out | ⚠️ 偶发 | 读大文件超时（如网络盘/慢盘），该 run 被跳过；可重跑或把 data 放本地 SSD。 |
| --- Statistics --- 下面没有输出 | ✅ 预期 | 当任一组样本数 &lt; 2（如只有 Low）时不做统计。 |
| Saved ... png | ✅ 预期 | 图正常生成；仅一组有数据时箱线图可能只显示一个箱子。 |

### English

| Message / behaviour | Expected? | Note |
|---------------------|-----------|------|
| Trials: 5 (Low: 5, High: 0) | ✅ Yes | With `--max-runs 5`, first 5 rows may all be Low; increase runs or run full for High. |
| Omitted/Limited annotation(s) | ✅ Yes | MNE ignores/trims out-of-range events; warnings suppressed. |
| boundary events / discontinuities | ✅ Yes | Normal for continuous data; theta uses full segment. |
| [Errno 60] Operation timed out | ⚠️ Occasional | I/O timeout; that run is skipped; retry or use local SSD. |
| No output under --- Statistics --- | ✅ Yes | Skipped when either group has &lt; 2 samples. |
| Saved ... png | ✅ Yes | Figures written; boxplot may show one group only if the other is empty. |

---

## 六、从 electrodes.tsv 自行生成区域映射 / Generate region mapping from electrodes.tsv

### 中文

任选一份 `*_electrodes.tsv`（所有被试同帽），按 z 划分即可，例如：

```python
with open("data/sub-020/ses-02/eeg/..._electrodes.tsv") as f:
    next(f)
    for line in f:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        name, z = parts[0], float(parts[3])
        if name == "E129":
            continue
        if z >= 5:    region = "frontal"
        elif z >= -2:  region = "central"
        else:         region = "parietal"
        print(f"{name},{region}")
```

### English

Use any single `*_electrodes.tsv`; assign by z as above (frontal z≥5, central -2≤z&lt;5, parietal z&lt;-2, skip E129). Example loop in Python is given in the Chinese block above.
