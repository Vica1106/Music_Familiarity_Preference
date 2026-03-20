# Music Familiarity & Theta EEG — Project Summary

This document summarizes the research question, electrode region mapping, analysis pipeline, results, generated figures, and how to run the scripts.

---

## 1. Research Question

**Does music familiarity affect theta-band (4–8 Hz) EEG activity?**

We compare **Low familiarity (ratings 1–2)** vs **High familiarity (rating 5)** theta power, with a focus on regional differences (frontal, central, parietal).

---

## 2. Results Summary

### Statistical Results (Full Dataset: Low n=51, High n=63)

| Region | t-test | Mann-Whitney U | Conclusion |
|--------|--------|----------------|------------|
| **Frontal** | t = −2.50, **p = 0.014** | U = 1027, **p = 0.001** | **Significant** ✓ |
| Central | t = −1.32, p = 0.189 | U = 1434, p = 0.327 | Not significant |
| Parietal | t = 0.20, p = 0.845 | U = 1313, p = 0.095 | Not significant |

### Key Findings

1. **Frontal theta power is significantly different between Low and High familiarity groups**
   - Both parametric (t-test) and non-parametric (Mann-Whitney U) tests show p < 0.05
   - The negative t-value indicates: **Low familiarity < High familiarity**
   - **Interpretation: Higher music familiarity is associated with greater frontal theta power**

2. **The effect is region-specific (frontal only)**
   - Central and parietal theta show no significant group differences
   - This supports a cognitive/memory interpretation rather than a general sensory effect

### Neuroscientific Interpretation

- Frontal theta is associated with **memory retrieval, attention, and cognitive engagement**
- Familiar music may activate memory networks (hippocampus–prefrontal circuits), leading to enhanced frontal theta
- The frontal-specific effect is consistent with the role of prefrontal cortex in recognition and familiarity processing

---

## 3. Generated Figures

### Main Analysis Figures (`results/`)

| Figure | Description |
|--------|-------------|
| `theta_power_vs_familiarity_group.png` | **Main result**: Boxplot of frontal theta power for Low vs High familiarity |
| `theta_power_by_region.png` | Three boxplots showing theta power by region (frontal/central/parietal) |
| `theta_topomap.png` | Brain topography of average theta power across all trials |
| `theta_topomap_by_familiarity_group.png` | Side-by-side topomaps: Low vs High familiarity |

### Exploratory Figures (`results/`)

| Figure | Description |
|--------|-------------|
| `explore_spectrogram_theta.png` | Time–frequency spectrogram (single channel, theta visible) |
| `explore_all_bands_over_time.png` | All five frequency bands (delta/theta/alpha/beta/gamma) power over time |
| `explore_theta_alpha_over_time.png` | Frontal theta and alpha power over time |
| `explore_theta_by_region_over_time.png` | Theta power over time by frontal/central/parietal |
| `explore_topomap_familiarity_low_vs_high.png` | Graph 7–style: Example theta topomaps for Low vs High trials (3 each) |

---

## 4. Electrode Region Mapping

**Data source**  
Each run has an `*_electrodes.tsv` file with EGI 128 channel positions (columns: `name`, `x`, `y`, `z`).

**Mapping rule (by z-coordinate)**  
Using **EGI 128 HCGSN** cap, where **z** encodes anterior–posterior/superior position:

| Region | Condition | Meaning |
|--------|-----------|---------|
| **frontal** | z ≥ 5 | Anterior/frontal |
| **central** | −2 ≤ z < 5 | Central |
| **parietal** | z < −2 | Posterior/parietal-occipital |

**E129** (Cz reference) is excluded from region averages. The file `electrode_regions.csv` is a pre-generated reference; analysis computes regions on the fly from each run's electrodes file.

---

## 5. How to Run

### Dependencies

```bash
pip install -r requirements.txt   # numpy, pandas, scipy, matplotlib, mne, tqdm
```

### Main Analysis (familiarity vs theta)

```bash
# Quick test on first N runs
python analysis_theta_familiarity.py --max-runs 5

# Full analysis (uses cached CSV if available)
python analysis_theta_familiarity.py

# Force recompute (ignore cache)
python analysis_theta_familiarity.py --recompute
```

**Outputs**: `results/theta_by_familiarity.csv`, boxplots, topomaps, statistics, and log file.

### Exploratory Analysis (event-based epochs, TFR, multiple plots)

```bash
python explore_theta_familiarity.py
python explore_theta_familiarity.py --sub 1 --ses 2
python explore_theta_familiarity.py --max-runs 10
```

---

## 6. Expected Behaviour and Common Messages

| Message / Behaviour | Expected? | Note |
|---------------------|-----------|------|
| `Found cached features: ...csv` | ✅ Yes | Using cached data; figures regenerated without recomputing EEG |
| Omitted/Limited annotation(s) | ✅ Yes | MNE ignores out-of-range events; warnings suppressed |
| boundary events / discontinuities | ✅ Yes | Normal for continuous data |
| [Errno 60] Operation timed out | ⚠️ Occasional | I/O timeout; run skipped; use local SSD |
| No statistics output | ✅ Yes | Skipped when either group has < 2 samples |

---

## 7. Conclusion Statement (for Report/Paper)

> **Higher music familiarity was associated with significantly greater frontal theta power (t = −2.50, p = 0.014; Mann–Whitney U = 1027, p = 0.001), while central and parietal theta showed no significant group differences. This frontal-specific effect suggests that music familiarity modulates theta oscillations in regions associated with memory retrieval and cognitive engagement.**

---
---

# 中文版 / Chinese Version

---

## 一、研究问题

**音乐熟悉度（Familiarity）是否会影响 theta 频段（4–8 Hz）的 EEG 活动？**

比较 **低熟悉度（1–2 分）** 与 **高熟悉度（5 分）** 两组的 theta 功率；重点关注脑区差异（额区、中央区、顶区）。

---

## 二、结果摘要

### 统计结果（全量数据：Low n=51, High n=63）

| 脑区 | t 检验 | Mann-Whitney U | 结论 |
|------|--------|----------------|------|
| **额区 Frontal** | t = −2.50, **p = 0.014** | U = 1027, **p = 0.001** | **显著差异** ✓ |
| 中央区 Central | t = −1.32, p = 0.189 | U = 1434, p = 0.327 | 无显著差异 |
| 顶区 Parietal | t = 0.20, p = 0.845 | U = 1313, p = 0.095 | 无显著差异 |

### 主要发现

1. **额区 theta 功率在 Low 与 High 熟悉度组之间存在显著差异**
   - 参数检验（t-test）和非参数检验（Mann-Whitney U）的 p 值均 < 0.05
   - **负的 t 值**意味着：**Low 熟悉度 < High 熟悉度**
   - **解释：越熟悉的音乐 → 额区 theta 功率越高**

2. **该效应具有脑区特异性（仅在额区显著）**
   - 中央区和顶区的 theta 功率在两组之间无显著差异
   - 支持认知/记忆机制的解释，而非单纯的感觉差异

### 神经科学解释

- **Frontal theta（额区 θ 波）** 在文献中通常与 **记忆提取、注意、认知加工** 相关
- 熟悉的音乐可能激活更多的记忆网络（如 hippocampus–prefrontal 回路），导致额区 theta 增强
- 这种脑区特异性支持前额叶在熟悉度识别中的作用

---

## 三、生成的图表

### 主分析图（`results/`）

| 文件名 | 说明 |
|--------|------|
| `theta_power_vs_familiarity_group.png` | **主结果图**：额区 theta 功率 × 熟悉度组别（Low/High）箱线图 |
| `theta_power_by_region.png` | 三个脑区（额/中央/顶）的 theta 功率箱线图 |
| `theta_topomap.png` | 全 trial 平均的 theta 功率脑地形图 |
| `theta_topomap_by_familiarity_group.png` | Low vs High 熟悉度的 theta 地形图（并排） |

### 探索性分析图（`results/`）

| 文件名 | 说明 |
|--------|------|
| `explore_spectrogram_theta.png` | 单通道时–频图（theta 可见） |
| `explore_all_bands_over_time.png` | 五频段（delta/theta/alpha/beta/gamma）功率随时间变化 |
| `explore_theta_alpha_over_time.png` | 额区 theta 与 alpha 功率随时间变化 |
| `explore_theta_by_region_over_time.png` | 额/中央/顶三个脑区的 theta 功率随时间变化 |
| `explore_topomap_familiarity_low_vs_high.png` | 类似 Graph 7：Low vs High 各 3 个 example trial 的 theta 地形图 |

---

## 四、电极区域映射

**数据来源**  
每个被试的 EEG 数据目录下有一份 `*_electrodes.tsv`，内容为 EGI 128 通道帽的电极坐标（列：`name`, `x`, `y`, `z`）。

**映射规则（按 z 坐标）**  
本数据集使用 **EGI 128 HCGSN** 电极帽，**z 轴**表示头部的"前–后/上"方向：

| 区域 | 条件 | 含义 |
|------|------|------|
| **frontal** | z ≥ 5 | 前部/额叶 |
| **central** | −2 ≤ z < 5 | 中部/中央区 |
| **parietal** | z < −2 | 后部/顶枕区 |

**E129** 为参考电极 (Cz)，不参与区域平均。`electrode_regions.csv` 为预生成的映射表，仅作参考；**实际分析时区域在代码里按每份 electrodes.tsv 的 z 实时计算**。

---

## 五、如何运行

### 依赖

```bash
pip install -r requirements.txt   # numpy, pandas, scipy, matplotlib, mne, tqdm
```

### 主分析（familiarity vs theta）

```bash
# 小数据快速测试（仅前 N 个 trial）
python analysis_theta_familiarity.py --max-runs 5

# 全量分析（有缓存 CSV 则直接画图）
python analysis_theta_familiarity.py

# 强制重算（忽略缓存）
python analysis_theta_familiarity.py --recompute
```

**输出**：`results/theta_by_familiarity.csv`、箱线图、脑地形图、统计结果、时间戳日志。

### 探索性分析（事件分段 + TFR + 多图）

```bash
python explore_theta_familiarity.py
python explore_theta_familiarity.py --sub 1 --ses 2
python explore_theta_familiarity.py --max-runs 10
```

---

## 六、运行时的预期行为与常见提示

| 现象 | 是否预期 | 说明 |
|------|----------|------|
| `Found cached features: ...csv` | ✅ 预期 | 使用缓存数据，不重算 EEG，只重画图 |
| Omitted/Limited annotation(s) | ✅ 预期 | MNE 忽略超出范围的事件 |
| boundary events / discontinuities | ✅ 预期 | 连续记录数据的正常情况 |
| [Errno 60] Operation timed out | ⚠️ 偶发 | 读大文件超时，该 run 被跳过；建议用本地 SSD |
| 统计部分无输出 | ✅ 预期 | 当任一组样本数 < 2 时不做统计 |

---

## 七、可用于报告/论文的结论

> **更高的音乐熟悉度与更强的额区 theta 功率显著相关（t = −2.50, p = 0.014; Mann–Whitney U = 1027, p = 0.001），而中央区和顶区 theta 无显著组间差异。这种额区特异性效应表明，音乐熟悉度调控了与记忆提取和认知加工相关脑区的 theta 振荡活动。**

---

## 八、从 electrodes.tsv 自行生成区域映射

任选一份 `*_electrodes.tsv`（所有被试同帽），按 z 划分：

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
        elif z >= -2: region = "central"
        else:         region = "parietal"
        print(f"{name},{region}")
```
