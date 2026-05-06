# Vision Analysis Worksheet

请对以下每张图片进行详细的学术图表分析（中文，≤300 字）。
分析时请结合提供的 Markdown 表格数据和页面上下文文本。

## 分析要求
1. 识别图表类型及其对比内容
2. 提取关键数值结果和趋势
3. 结合论文上下文解释研究发现的意义

## 输出格式
请将全部 29 张图片的分析结果保存为如下 JSON 文件，
命名为 `vision_answers.json` 放入 `dataset/answers/` 目录：

```json
[
  {"image_id": "vis_01", "analysis": "该表格展示了..."},
  {"image_id": "vis_02", "analysis": "..."},
  ...
]
```

---

## Image vis_01 (`vis_01.jpg`)

- **Paper ID**: `Object Detection in Autonomous Vehicles under Adverse
Weather: A Review of Traditional and Deep
Learning Approaches.pdf`
- **Page**: 2
- **Context Text** (前 500 字符): Algorithms 2024, 17, 103
2 of 36
autonomous vehicles (AVs). AVs, especially self-driving cars, have generated a great deal
of interest in the fields of intelligent transportation, computer vision, and artificial intel-
ligence [2]. No other invention since the creation of vehicles has had such a dramatic
impact on the automotive industry as self-driving vehicles. As per the forecasts presented
in Figure 1 [3], the market for self-driving cars is expected to grow from 23.80 million
units in 2022 

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Col0   | Col1   | Col2   | Col3   | Col4   | Col5   | Col6   | Col7   | Col8   | Col9   | Col10   | Col11   | Col12   | Col13   | Col14   | Col15   | Col16   | Col17   | Col18   | Col19   | Col20   |
|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|
|        |        |        |        |        |        |        |        |        |        |         |         |         |         |         |         |         |         |         |         |         |
|        |        |        |        |        |        |        |        |        |        |         |         |         |         |         |         |         |         |         |         |         |
|        |        |        |        |        |        |        |        |        |        |         |         |         |         |         |         |         |         |         |         |         |
|        |        |        |        |        |        |        |        |        |        |         |         |         |         |         |         |         |         |         |         |         |
|        |        |        |        |        |        |        |        |        |        |         |         |         |         |         |         |         |         |         |         |         |
|        |        |        |        |        |        |        |        |        |        |         |         |         |         |         |         |         |         |         |         |         |
|        |        |        |        |        |        |        |        |        |        |         |         |         |         |         |         |         |         |         |         |         |
|        |        |        |        |        |        |        |        |        |        |         |         |         |         |         |         |         |         |         |         |         |
|        |        |        |        |        |        |        |        |        |        |         |         |         |         |         |         |         |         |         |         |         |
|        |        |        |        |        |        |        |        |        |        |         |         |         |         |         |         |         |         |         |         |         |
|        |        |        |        |        |        |        |        |        |        |         |         |         |         |         |         |         |         |         |         |         |
|        |        |        |        |        |        |        |        |        |        |         |         |         |         |         |         |         |         |         |         |         |
|        |        |        |        |        |        |        |        |        |        |         |         |         |         |         |         |         |         |         |         |         |
|        |        |        |        |        |        |        |        |        |        |         |         |         |         |         |         |         |         |         |         |         |
|        |        |        |        |        |        |        |        |        |        |         |         |         |         |         |         |         |         |         |         |         |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_02 (`vis_02.jpg`)

- **Paper ID**: `Object Detection in Autonomous Vehicles under Adverse
Weather: A Review of Traditional and Deep
Learning Approaches.pdf`
- **Page**: 4
- **Context Text** (前 500 字符): Algorithms 2024, 17, 103
4 of 36
Section1
Introduction
Sensor
Technologies
Section2 
RelatedSurvey
Actuators
Section3 
OverviewofAVs
O
R
G
A
N
I
Z
A
T
I
O
N
Perception
Section4
Overview of OD
Planningand
Control
Section5
Overviewof 
Traditional / 
DLApproaches
Challengesfor
AVs under
Adverse 
weather
Section6
Previous work in
OD
Road Lane
Vehicle 
Detection
Pedestrian
Section7
Discussion /
Limitation/Future
Detection
Detection
Work
Section 8
Conclusion


[TABLE_0_PLACEHOLDER]


2. Related Survey

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Col0           | Col1   | Col2   |
|:---------------|:-------|:-------|
| Figure 3. Pape |        |        |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_03 (`vis_03.jpg`)

- **Paper ID**: `Nakamura_Active_Domain_Adaptation_with_False_Negative_Prediction_for_Object_Detection_CVPR_2024_paper.pdf`
- **Page**: 7
- **Context Text** (前 500 字符): Table 1. Comparison with the state-of-the-art UDA and ADA
methods. Even with only 1% of the labeling budget (Ours (1%)),
the performance of our proposed method exceeded that of almost
all conventional methods in terms of accuracy. Furthermore, our
proposed method achieved the performance nearly equivalent to
that of Oracle while using 5% of the labeling budget (Ours (5%)).
AADA† represents an evaluation through re-implementation.
4.1.2
Implementation Details
Network Architecture. We used Faster-

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| 50                  | Col1   | Col2   | Col3   | Col4   | Col5   | Col6   | Col7        | Col8   |
| 45 (%)              |        |        |        |        |        |        |             |        |
| Random              |        |        |        |        |        |        |             |        |
| mAP                 |        |        |        |        |        |        |             |        |
| MeanEntropy         |        |        |        |        |        |        |             |        |
| 40 CoreSet          |        |        |        |        |        |        |             |        |
| ActiveTeacher       |        |        |        |        |        |        |             |        |
| AADA                |        |        |        |        |        |        |             |        |
| 35 Ours             |        |        |        |        |        |        |             |        |
| Oracle              |        |        |        |        |        |        |             |        |
| 0.1 0.3 0.5 1 3 5   |        |        |        |        |        |        |             |        |
| Budget (%)          |        |        |        |        |        |        |             |        |
|:--------------------|:-------|:-------|:-------|:-------|:-------|:-------|:------------|:-------|
|                     |        |        |        |        |        |        |             | y      |
|                     |        |        |        |        |        |        | Random      |        |
|                     |        |        |        |        |        |        | MeanEntrop  |        |
|                     |        |        |        |        |        |        | CoreSet     | er     |
|                     |        |        |        |        |        |        | ActiveTeach |        |
|                     |        |        |        |        |        |        | AADA        |        |
|                     |        |        |        |        |        |        | Ours        |        |
|                     |        |        |        |        |        |        | Oracle      |        |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_04 (`vis_04.jpg`)

- **Paper ID**: `Chen_Tokenize_Image_Patches_Global_Context_Fusion_for_Effective_Haze_Removal_CVPR_2025_paper.pdf`
- **Page**: 8
- **Context Text** (前 500 字符): Attribution Maps
Dehazed Results
Attribution Maps
Dehazed Results
8.278 / 0.6453
8.278 / 0.6453
PSNR / SSIM
(1-a) Input
PSNR / SSIM
(1-a) Input
PSNR / SSIM
(1-b) GT
(1-b) GT
(1-c) 4KDehazing (Slicing)
17.20 / 0.8816
(1-d) 4KDehazing (Direct)
14.63 / 0.8680
(1-e) Dehamer
21.63 / 0.8874
(1-f) C2PNet
20.79 / 0.9464
(1-g) DehazeFormer-b
 20.40 / 0.9398
(1-h) MB-TaylorFormer
21.43 / 0.9543
(1-i) ConvIR-b
19.45 / 0.9428
(1-j) MixDehazeNet-b
21.42 / 0.9506
(1-k) DEA-Net
20.29 / 0.8672
(1-l) DehazeXL(ou

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Backbone   | Bottleneck   | Metrics   | Col3   | Col4    |
| Type       | Depth        |           |        |         |
|:-----------|:-------------|:----------|:-------|:--------|
|            |              | PSNR      | SSIM   | Time(s) |
| Swin-T     | 1            | 31.61     | 0.9719 | 4.511   |
|            | 2            | 32.35     | 0.9863 | 4.617   |
|            | 4            | 32.40     | 0.9857 | 4.810   |
| Swin-S     | 1            | 31.89     | 0.9759 | 5.880   |
|            | 2            | 32.58     | 0.9871 | 5.967   |
|            | 4            | 32.76     | 0.9870 | 6.102   |
| Swin-B     | 1            | 32.10     | 0.9792 | 9.066   |
|            | 2            | 32.77     | 0.9879 | 9.183   |
|            | 4            | 33.06     | 0.9894 | 9.526   |
| Swin-L     | 1            | 32.93     | 0.9877 | 17.37   |
|            | 2            | 32.98     | 0.9885 | 17.64   |
|            | 4            | 33.30     | 0.9911 | 18.25   |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_05 (`vis_05.jpg`)

- **Paper ID**: `Object Detection in Autonomous Vehicles under Adverse
Weather: A Review of Traditional and Deep
Learning Approaches.pdf`
- **Page**: 27
- **Context Text** (前 500 字符): Algorithms 2024, 17, 103
27 of 36


[TABLE_0_PLACEHOLDER]



### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Col0                                                                 | detectors are frequently used for road lanes compared with single-stage d   | etectors.   |
|                                                                      | Vehicles Pedestrians Road Lanes                                             |             |
|                                                                      | 32% 40%                                                                     |             |
|                                                                      | 28%                                                                         |             |
|:---------------------------------------------------------------------|:----------------------------------------------------------------------------|:------------|
| Figure 13. Overall percentage distribution of papers.                | Figure 13. Overall percentage distribution of papers.                       |             |
| 25                                                                   |                                                                             |             |
| 20                                                                   |                                                                             |             |
| Papers                                                               |                                                                             |             |
| 15                                                                   |                                                                             |             |
| of                                                                   |                                                                             |             |
| 10                                                                   |                                                                             |             |
| Number                                                               |                                                                             |             |
| 5                                                                    |                                                                             |             |
| 0                                                                    |                                                                             |             |
| Traditional Approaches DL Approaches                                 |                                                                             |             |
| Vehicles Pedestrians Road Lanes                                      |                                                                             |             |
| Figure 14. Distribution of papers for traditional and DL approaches. |                                                                             |             |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_06 (`vis_06.jpg`)

- **Paper ID**: `Domain_Adaptation_based_Object_Detection_for_Autonomous_Driving_in_Foggy_and_Rainy_Weather.pdf`
- **Page**: 7
- **Context Text** (前 500 字符): This article has been accepted for publication in IEEE Transactions on Intelligent Vehicles. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TIV.2024.3419689
7
TABLE I
ADAPTATION FROM CLEAR TO FOGGY: CITYSCAPES→FOGGY CITYSCAPES EXPERIMENT. NOTE THAT ORACLE REPRESENTS THE FASTER R-CNN
TRAINED ON FOGGY CITYSCAPE TRAINING SET WITH ALL LABELS. THE BEST PERFORMANCE IS BOLD AND THE SECOND BEST IS UNDERLIN

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| MCAR-ECCV’2020 [51]        | 44.1 36.6 43.9 37.4 32.0 42.1 43.4 31.3   | 38.8   |
| MTOR-CVPR-2019 [70]        | 38.6 35.6 44.0 28.3 30.6 41.4 40.6 21.9   | 35.1   |
| DA-Faster-CVPR’2018 [23]   | 49.8 39.0 53.0 28.9 35.7 45.2 45.4 30.9   | 41.0   |
| GPA-CVPR’2020 [54]         | 45.7 38.7 54.1 32.4 32.9 46.7 41.1 24.7   | 39.5   |
| RPN-PR-CVPR’2021 [53]      | 43.6 36.8 50.5 29.7 33.3 45.6 42.0 30.4   | 39.0   |
| UaDAN-TMM’2021 [26]        | 49.4 38.9 53.6 32.3 36.5 46.1 42.7 28.9   | 41.1   |
| HTCN-CVPR’2020 [71]        | 47.4 37.1 47.9 32.3 33.2 47.5 40.9 31.6   | 39.8   |
| SAPN-ECCV’2020 [72]        | 46.8 40.7 59.8 30.4 40.8 46.7 37.5 24.3   | 40.9   |
| MeGA-CDA-CVPR’2021 [73]    | 49.2 39.0 52.4 34.5 37.7 49.0 46.9 25.4   | 41.8   |
| UMT-CVPR2021 [74]          | 56.6 37.3 48.6 30.4 33.0 46.7 46.8 34.1   | 41.7   |
| SCAN-AAAI’2022 [75]        | 48.6 37.3 57.3 31.0 41.7 43.9 48.7 28.7   | 42.1   |
| ParaUDA-TITS’2022 [76]     | 48.3 37.6 52.5 33.5 36.7 46.7 45.9 32.3   | 41.7   |
| ConfMix-WACV’2023 [77]     | 45.8 33.5 62.6 28.6 45.0 43.4 40.0 27.3   | 40.8   |
| SDAYOLO-TIV’2023 [78]      | 40.5 37.3 61.9 24.4 42.6 42.1 39.5 23.5   | 39.0   |
| MS-DAYOLO-TIP’2023 [79]    | 51.0 36.0 56.5 27.5 39.6 46.5 45.9 28.9   | 41.5   |
|----------------------------|-------------------------------------------|--------|

**TABLE_1**:

| Ours w/o Auxiliary Domain   | 48.4 36.7 53.5 26.1 36.1 45.9 39.1 29.3   | 40.2   |
| Ours                        | 51.2 39.1 54.3 31.6 36.5 46.7 48.7 30.3   | 42.3   |
| Ours+                       | 48.7 41.6 55.8 33.3 36.5 49.1 51.3 30.0   | 43.4   |
|-----------------------------|-------------------------------------------|--------|

**TABLE_2**:

| Faster R-CNN (source only)   | 46.3 26.0 54.8 25.8 34.7 35.9 26.9 23.9   | 34.3   |
| DA-Faster-CVPR’2018 [23]     | 54.6 33.8 59.6 32.4 38.2 41.6 42.9 33.8   | 42.1   |
| MS-DAYOLO-TIP’2023 [79]      | 60.3 34.4 67.7 31.9 47.2 47.4 31.9 30.7   | 44.0   |
|------------------------------|-------------------------------------------|--------|

**TABLE_3**:

| Ours w/o Auxiliary Domain   | 55.5 35.3 60.0 32.8 38.3 22.1 49.3 33.1   | 43.4   |
| Ours                        | 60.0 35.3 60.6 33.8 38.8 42.9 52.4 36.3   | 45.0   |
| Ours+                       | 59.7 39.0 61.7 34.4 40.2 47.0 47.2 38.8   | 46.0   |
|-----------------------------|-------------------------------------------|--------|

**TABLE_4**:

| Methods                   | Img Obj AdvGRL Reg DMP   | Foggy mAP Rainy mAP   |
|:--------------------------|:-------------------------|:----------------------|
| Source only               | ✓                        | 23.41 34.35           |
| w/ DMP                    | ✓                        | 24.54 35.90           |
| img w/ GRL                | ✓                        | 38.10 36.41           |
| obj w/ GRL                | ✓ ✓                      | 38.02 37.92           |
| img+obj w/ GRL (Baseline) | ✓ ✓ ✓                    | 38.43 41.02           |
| img+obj w/AdvGRL          | ✓ ✓ ✓                    | 40.23 43.44           |
| img+obj+ Reg w/ GRL       | ✓ ✓ ✓ ✓                  | 41.97 44.44           |
| img+obj+Reg w/ AdvGRL     | ✓ ✓ ✓ ✓ ✓                | 42.34 45.07           |
| img+obj+Reg+DMP w/ AdvGRL |                          | 43.42 46.04           |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_07 (`vis_07.jpg`)

- **Paper ID**: `Multi-Level_Feature_Fusion_Based_UAV_Object_Detection_Method_Under_Foggy_Weather.pdf`
- **Page**: 5
- **Context Text** (前 500 字符): 

[TABLE_0_PLACEHOLDER]




[TABLE_1_PLACEHOLDER]


a. Precision(%).b.Recall(%) 
In particular, we observe that the enhanced modules have 
significantly boosted the accuracy of detecting small targets 
under foggy conditions from an aerial perspective. Among the 
three individual improvements to the baseline, the FDM had 
the most substantial impact on performance, yielding a 0.9% 
increase on mAP0.5 compared to the baseline. This finding 
underscores the critical role of image preprocessing in 

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Detection     |   P(%)a |   R(%)b |   mAP0.5(%) |   mAP0.5:0.9(%) |
| algorithm     |         |         |             |                 |
|:--------------|--------:|--------:|------------:|----------------:|
| IA-YOLO       |    36.3 |    10.9 |        12.9 |             8   |
| Fog Guard     |    37.6 |    11.1 |        13.1 |             8.4 |
| DE-YOLO       |    36.4 |    10.9 |        13   |             8   |
| YOLOv10s      |    35.2 |    11   |        12.8 |             7.7 |
| DU-YOLO(ours) |    37.9 |    11.2 |        13.4 |             8.6 |

**TABLE_1**:

| Detection       |   P(%)a |   R(%)b |   mAP0.5(%) |   mAP0.5:0.9(%) |
| algorithm       |         |         |             |                 |
|:----------------|--------:|--------:|------------:|----------------:|
| YOLOv10s+FDM+   |    51.6 |    37.4 |        39.6 |            24.2 |
| MLM             |         |         |             |                 |
| YOLOv10s+FDM+   |    52   |    38.3 |        41.6 |            24.6 |
| MLM+Loss (ours) |         |         |             |                 |

**TABLE_2**:

| Detection     |   P(%)a |   R(%)b |   mAP0.5(%) |   mAP0.5:0.9(%) |
| algorithm     |         |         |             |                 |
|:--------------|--------:|--------:|------------:|----------------:|
| Faster R-CNN  |    28.2 |     5.3 |         8.6 |             4.9 |
| IA-YOLO       |    32.6 |     6.5 |         9.5 |             5.8 |
| Fog Guard     |    33.3 |     6.9 |         9.8 |             5.9 |
| DE-YOLO       |    32.9 |     6.8 |         9.5 |             6.1 |
| YOLOv10s      |    32.4 |     6.7 |         9.2 |             5.2 |
| DU-YOLO(ours) |    33.9 |     6.9 |        10   |             6.3 |

**TABLE_3**:

| Detection     |   P(%)a |   R(%)b |   mAP0.5(%) |   mAP0.5:0.9(%) |
| algorithm     |         |         |             |                 |
|:--------------|--------:|--------:|------------:|----------------:|
| YOLOv10s      |    50.8 |    35.3 |        36.7 |            21.9 |
| YOLOv10s+FDM  |    51.3 |    35.4 |        37.6 |            23.1 |
| YOLOv10s+MLM  |    51   |    34.9 |        37.3 |            22.4 |
| YOLOv10s+Loss |    50.8 |    35.4 |        36.9 |            22   |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_08 (`vis_08.jpg`)

- **Paper ID**: `Object Detection in Autonomous Vehicles under Adverse
Weather: A Review of Traditional and Deep
Learning Approaches.pdf`
- **Page**: 28
- **Context Text** (前 500 字符): 

[TABLE_0_PLACEHOLDER]


In order to improve OD, Faster R-CNN was integrated into the RPN. This resulted
in shorter evaluation times, but it also required a significant amount of processing power
and had difficulties maintaining consistent performance for small and variously scaled
objects.YOLOv8 is designed to cater to specific requirements. The YOLOv8 structure
comprises backbone, neck, and head modules, enhancing feature extraction with C2f and
employing the PAN feature fusion technique. It 

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Algorithms 2024, 17,   | 103 28 of 36                                                         |
|:-----------------------|:---------------------------------------------------------------------|
|                        | 28 of 36                                                             |
|                        | One-Stage detectors Two-Stage detectors Both detectors               |
|                        | 16                                                                   |
|                        | 14 2                                                                 |
|                        | 12                                                                   |
|                        | Papers                                                               |
|                        | 4                                                                    |
|                        | 10                                                                   |
|                        | 8 of                                                                 |
|                        | 2 Number                                                             |
|                        | 6                                                                    |
|                        | 9 3                                                                  |
|                        | 4                                                                    |
|                        | 1                                                                    |
|                        | 2                                                                    |
|                        | 3                                                                    |
|                        | 2                                                                    |
|                        | 0                                                                    |
|                        | Vehicles Pedestrians Road Lanes                                      |
|                        | Figure 15. Paper distribution for one-stage and two-stage detectors. |

**TABLE_1**:

|   Col0 | Col1   |   Col2 | Col3   |   Col4 |
|-------:|:-------|-------:|:-------|-------:|
|        |        |        |        |        |
|      2 |        |        |        |        |
|        |        |        |        |        |
|        |        |        |        |        |
|        |        |        |        |        |
|        |        |        |        |        |
|        |        |        |        |        |
|      4 |        |        |        |        |
|        |        |        |        |        |
|        |        |        |        |        |
|        |        |        |        |        |
|        |        |        |        |        |
|        |        |        |        |        |
|        |        |        |        |        |
|        |        |        |        |        |
|        |        |        |        |        |
|        |        |      2 |        |        |
|        |        |        |        |        |
|        |        |        |        |        |
|        |        |        |        |        |
|        |        |        |        |        |
|        |        |        |        |        |
|      9 |        |      3 |        |        |
|        |        |        |        |        |
|        |        |        |        |        |
|        |        |        |        |        |
|        |        |        |        |      1 |
|        |        |        |        |        |
|        |        |        |        |        |
|        |        |      3 |        |        |
|        |        |        |        |      2 |
|        |        |        |        |        |
|        |        |        |        |        |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_09 (`vis_09.jpg`)

- **Paper ID**: `Chen_Tokenize_Image_Patches_Global_Context_Fusion_for_Effective_Haze_Removal_CVPR_2025_paper.pdf`
- **Page**: 5
- **Context Text** (前 500 字符): Integration Path
Hazy image
Baseline
ing all tokens instead of concurrently. This strategy allows
us to achieve signiﬁcantly lower memory usage at the cost
of slightly increased processing time.
3.2. Dehazing Attribution Map
Inspired by the Integrated Gradients (IG) method [42]
and the Local Attribution Map [13], we propose the De-
hazing Attribution Map to enhance the interpretability of
our model. Let F : Rhw →Rhw represent a dehaz-
ing network. To quantify the dehazing effect, we utilize a
pi

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Dataset    | 4KID O-HAZY 8KDehaze                             |
|:-----------|:-------------------------------------------------|
| Quantity   | 15606 45 10000                                   |
| Image Size | 1286 × 947                                       |
|            | 3840 × 2160 to 8192 × 8192                       |
|            | 5436 × 3612                                      |
| Source     | Video Frames Commercial Camera Aerial Images     |
| Content    | Urban, Farmland,                                 |
|            | Urban Streets Parks, Suburban Mountains, Desert, |
|            | Coastlines, Rivers                               |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_10 (`vis_10.jpg`)

- **Paper ID**: `CoDerainNet: Collaborative Deraining Network for
Drone-view Object Detection in Rainy Weather Conditions.pdf`
- **Page**: 8
- **Context Text** (前 500 字符): Remote Sens. 2023, 15, 1487
8 of 21
multi-scale features. TFB is a target-focusing block, which focuses the attention on RoIs to
suppress background noise.
Fine-grained Target Focusing Module
C5
P5
FiFo
Representation


[TABLE_1_PLACEHOLDER]




[TABLE_0_PLACEHOLDER]


P3
FiFA
C3
TFB
P2
Blocking 
Background
C2
Figure 3. The details of the Fine-grained Target-focusing module.
A regression loss for objects’ locations and a classiﬁcation loss for objects’ categories
are used to optimize DroneDet Su

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Col0   | Col1   | Col2   |
|:-------|:-------|:-------|
|        |        |        |
|        |        |        |

**TABLE_1**:

| Col0   | Col1   | Col2   |
|:-------|:-------|:-------|
|        |        |        |
|        |        |        |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_11 (`vis_11.jpg`)

- **Paper ID**: `Chen_Tokenize_Image_Patches_Global_Context_Fusion_for_Effective_Haze_Removal_CVPR_2025_paper.pdf`
- **Page**: 7
- **Context Text** (前 500 字符): (f) DehazeFormer-b
(e) C2PNet
(d) Dehamer
(c) 4KDehazing (Direct)
(b) 4KDehazing (Slicing)
(a) Input
(f) DehazeFormer-b
(e) C2PNet
(d) Dehamer
(c) 4KDehazing (Direct)
(b) 4KDehazing (Slicing)
(a) Input
19.66 / 0.7092
18.50 / 0.7141
17.57 / 0.7062
16.30 / 0.5339
17.81 / 0.6570
13.02 / 0.5657
19.66 / 0.7092
18.50 / 0.7141
17.57 / 0.7062
16.30 / 0.5339
17.81 / 0.6570
13.02 / 0.5657
(g) MB-TaylorFormer
(h) ConvIR-b
(i) MixDehazeNet-b
(j) DEA-Net
(k) DehazeXL(ours)
(l) GT
(g) MB-TaylorFormer
(h) Conv

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Method               | Venue&Year   | 8KDehaze   | Col3   | Col4    | 4KID [62]   | Col6   | Col7    | O-HAZY [2]   | Col9   | Col10   |
|:---------------------|:-------------|:-----------|:-------|:--------|:------------|:-------|:--------|:-------------|:-------|:--------|
|                      |              | PSNR       | SSIM   | Time(s) | PSNR        | SSIM   | Time(s) | PSNR         | SSIM   | Time(s) |
| 4KDehazing (Slicing) | CVPR2021     | 25.81      | 0.9569 | 6.682   | 19.97       | 0.8624 | 1.04    | 18.73        | 0.6726 | 1.31    |
| 4KDehazing (Direct)  | CVPR2021     | 20.41      | 0.8664 | 1.350   | 18.68       | 0.7424 | 0.19    | 19.3         | 0.6426 | 0.27    |
| Dehamer              | CVPR2022     | 25.92      | 0.9373 | 6.614   | 21.24       | 0.8795 | 1.03    | 19.59        | 0.7134 | 1.30    |
| C2PNet               | CVPR2023     | 26.17      | 0.9669 | 43.269  | 18.14       | 0.8299 | 6.76    | 20.29        | 0.7113 | 8.51    |
| DehazeFormer-s       | TIP2023      | 26.68      | 0.9729 | 7.469   | 20.83       | 0.8763 | 1.17    | 19.86        | 0.7116 | 1.47    |
| DehazeFormer-b       | TIP2023      | 26.83      | 0.9657 | 15.013  | 21.25       | 0.8843 | 2.35    | 20.22        | 0.7173 | 2.95    |
| MB-TaylorFormer      | ICCV2023     | 26.41      | 0.9668 | 120.540 | 18.63       | 0.8497 | 18.83   | 19.57        | 0.7104 | 23.71   |
| ConvIR-s             | TPAMI2024    | 25.11      | 0.9599 | 6.661   | 20.66       | 0.8696 | 1.04    | 18.83        | 0.7095 | 1.31    |
| ConvIR-b             | TPAMI2024    | 26.93      | 0.9775 | 8.709   | 21.92       | 0.888  | 1.36    | 19.61        | 0.7199 | 1.71    |
| MixDehazeNet-s       | IJCNN2024    | 20.99      | 0.8934 | 6.563   | 21.25       | 0.8817 | 1.03    | 19.09        | 0.7165 | 1.29    |
| MixDehazeNet-b       | IJCNN2024    | 23.16      | 0.9284 | 13.154  | 23.22       | 0.9063 | 2.06    | 20.67        | 0.7293 | 2.59    |
| DEA-Net              | TIP2024      | 25.89      | 0.9329 | 7.402   | 20.83       | 0.8834 | 1.16    | 20.01        | 0.6988 | 1.46    |
| DehazeXL             |              | 32.35      | 0.9863 | 4.617   | 26.62       | 0.9073 | 0.59    | 21.49        | 0.7348 | 0.86    |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_12 (`vis_12.jpg`)

- **Paper ID**: `Assessment_of_Efficient_and_Cost-Effective_Vehicle_Detection_in_Foggy_Weather.pdf`
- **Page**: 4
- **Context Text** (前 500 字符): 2024 18th International Conference on Open Source Systems and Technologies (ICOSST)
TABLE I
EXPERIMENTAL PARAMETERS (TRANING AND VALIDATION)


[TABLE_0_PLACEHOLDER]


In the third set of experiments, YOLO-V11m performance
is evaluated. The mAP50 score of car detection is 89.1 %
on the DAWN dataset and 68.7 % on the FD dataset. The
mAP50 score of bus detection is 33.2 % on the DAWN
dataset and 35.6 % on the FD dataset. The mAP50 score of
truck detection is 69.8 % on the DAWN dataset and 38.6 %
on

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Parameter(s)                    | Value(s)                                                             |
|:--------------------------------|:---------------------------------------------------------------------|
| YOLO-V11 Models                 | YOLO-V11n, YOLO-V11s, YOLO-V11m, YOLO-V11l                           |
| Datasets                        | DAWN, FD                                                             |
| Classes                         | 4 (Buses, Cars, Motorcycles, Trucks)                                 |
| Python and PyTorch Version      | Python-3.10.12, Torch-2.5.0+cu121 with CUDA enabled                  |
| Ultralytics Version             | 8.3.27                                                               |
| Hardware and GPU Specifications | 2 CPUs, Tesla T4 GPU (15GB), Systems RAM 12.7GB, Disk Space 112.6 GB |
| Experimental Platform           | Google Colab                                                         |
| Image Size                      | 640 train, 640 validation                                            |
| Batch Size (Traning)            | 16                                                                   |
| Batch Size (Validation)         | 1                                                                    |
| Training Epochs                 | 300                                                                  |
| Data Loaders/Workers            | 2                                                                    |
| Hyperparameters                 | Default                                                              |
| Optimizer                       | AdamW(lr=0.00125, momentum=0.9)                                      |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_13 (`vis_13.jpg`)

- **Paper ID**: `AdWeatherNet_Adverse_Weather_Denoising_with_Point_Cloud_Spatiotemporal_Attention.pdf`
- **Page**: 3
- **Context Text** (前 500 字符): Inputt-1
Spatial Encoder
Ft-1
N4
Clean Outputt
Semantic
Temporal-
differential 
Attention Block
Outlier 
Removal
Share weights
Inputt
Block
U-Net
Spatial Encoder
Ft
N4
(a) The framework of AdWeatherNet.
Density map
K
D
SubMConv3d


[TABLE_0_PLACEHOLDER]


Residual Block
Max pooling
(kernel=1)
Conv Layer
Sigmoid
A
Ft-1
Input
F
Voxelization
Ft
-
-
Variation computation ⦿Point-wise multiplication +
Concat
K Kernel size calculation in a voxel
D
Density prediction model
(c) Temporal-differential Atte

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Max pooling   | SubMConv3d (kernel=1)   | Softmax   |
|---------------|-------------------------|-----------|


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_14 (`vis_14.jpg`)

- **Paper ID**: `CoDerainNet: Collaborative Deraining Network for
Drone-view Object Detection in Rainy Weather Conditions.pdf`
- **Page**: 6
- **Context Text** (前 500 字符): Remote Sens. 2023, 15, 1487
6 of 21
Deraining Subnetwork
Deraining
Feature Extractor
Rain-free Image Generation Module
Reshape
DeConv3
DeConv1


[TABLE_0_PLACEHOLDER]


Conv1
Conv2
Deraining Loss
Reshape
Cat
Features
Reshape
Output: Rain-free Image
ColTeaching
CC
C5
P5
Feature Pyramid 
Network
CC
Input: Rainy Image
Conv1
Conv2
Fine-grained
Target Focusing
ConvN
Conv3
Detection Loss
.......
C3
P3
CC
Clean Features
P2
C2
Detection 
Feature Extractor
DroneDET Subnetwork
Figure 2. The architecture o

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| DeConv2   | Col1   |
|:----------|:-------|
|           |        |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_15 (`vis_15.jpg`)

- **Paper ID**: `Object Detection in Autonomous Vehicles under Adverse
Weather: A Review of Traditional and Deep
Learning Approaches.pdf`
- **Page**: 13
- **Context Text** (前 500 字符): Algorithms 2024, 17, 103
13 of 36
3.
Road Lane Detection: Road lane detection is the ability of AVs to identify and under-
stand the position and orientation of road lanes. This information is essential for the
vehicle to navigate within its designated lane, follow traffic rules, and make correct
turns, ensuring a smooth and safe driving experience.


[TABLE_2_PLACEHOLDER]




[TABLE_0_PLACEHOLDER]




[TABLE_3_PLACEHOLDER]


Figure 7. The basic framework for object detection systems.
5. Overvie

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| (aa)) Sensor   | Col1   |
|----------------|--------|

**TABLE_1**:

| Col0   | (bb)) ROI   |
|--------|-------------|

**TABLE_2**:

| 1 2 3   | Col1                  |
|:--------|:----------------------|
| 1 2 3   |                       |
| (cc     | )) Feature Extraction |

**TABLE_3**:

| Col0   | (dd)) Classification   | Col2   |
|--------|------------------------|--------|


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_16 (`vis_16.jpg`)

- **Paper ID**: `Domain_Adaptation_based_Object_Detection_for_Autonomous_Driving_in_Foggy_and_Rainy_Weather.pdf`
- **Page**: 9
- **Context Text** (前 500 字符): This article has been accepted for publication in IEEE Transactions on Intelligent Vehicles. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TIV.2024.3419689
9
(a)
(b)
(c)
(d)
Fig. 8.
Feature distribution visualization by t-SNE [80] before and after domain adaptation. Clear to Foggy Adaptation: (a) original distribution before
adaptation, (b) aligned distribution after the proposed adaptation. Clear

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Foggy / Rainy   | Methodologies      | Cbus Cbicycle Ccar Cmcycle Cperson Crider Ctrain Ctruck                                         | mAP         |
|:----------------|:-------------------|:------------------------------------------------------------------------------------------------|:------------|
| Large           | Source only        | 27.1 / 46.3 28.3 / 26.0 32.8 / 54.8 18.4 / 25.8 28.6 / 34.7 32.2 / 35.9 4.9 / 26.9 14.7 / 23.9  | 23.4 / 34.3 |
|                 | Baseline           | 45.4 / 49.5 36.7 / 32.6 53.5 / 58.3 26.0 / 31.4 36.1 / 36.1 45.9 / 41.4 37.1 / 43.4 26.3 / 35.1 | 38.4 / 41.0 |
|                 | Proposed DA Method | 51.2 / 60.0 39.1 / 35.3 54.3 / 60.6 31.6 / 33.8 36.5 / 38.8 46.7 / 42.9 48.7 / 52.4 30.3 / 36.3 | 42.3 / 45.0 |
| Medium          | Source only        | 40.6 / 47.1 36.9 / 26.9 48.9 / 54.6 27.3 / 23.9 37.9 / 34.0 43.2 / 38.6 40.4 / 30.8 22.4 / 30.7 | 37.2 / 35.8 |
|                 | Baseline           | 52.0 / 47.3 40.5 / 33.3 58.9 / 58.2 31.7 / 27.9 40.8 / 35.8 50.0 / 44.3 39.8 / 35.1 29.9 / 36.4 | 42.9 / 39.8 |
|                 | Proposed DA Method | 52.6 / 58.8 42.6 / 37.1 59.3 / 60.0 32.1 / 30.9 41.2 / 38.6 47.5 / 45.1 48.8 / 41.0 32.5 / 39.7 | 44.6 / 43.9 |
| Small           | Source only        | 49.2 / 43.8 40.9 / 29.6 55.7 / 55.7 33.1 / 24.1 41.0 / 35.7 47.0 / 37.5 43.1 / 38.7 28.4 / 23.3 | 42.3 / 36.0 |
|                 | Baseline           | 54.1 / 42.9 40.6 / 34.5 60.3 / 59.0 32.5 / 30.7 42.0 / 36.7 51.0 / 43.7 49.3 / 48.1 31.9 / 36.1 | 45.2 / 41.5 |
|                 | Proposed DA Method | 52.9 / 53.6 43.1 / 38.3 60.6 / 61.3 36.3 / 33.6 42.7 / 39.4 49.4 / 42.8 54.8 / 51.4 36.5 / 35.7 | 47.0 / 44.5 |

**TABLE_1**:

| Methodologies   | Car AP   |
|-----------------|----------|

**TABLE_2**:

| MAF-ICCV’2019 [55]       | 72.10   |
| SWDA-CVPR’2019 [50]      | 71.00   |
| ATF-ECCV’2020 [81]       | 73.50   |
| ART-CVPR’2020 [28]       | 73.60   |
| GPA-CVPR’2020 [54]       | 65.36   |
| SGA-TMM’2021 [82]        | 72.02   |
| UIT-ESwA’2022 [83]       | 73.70   |
| ParaUDA-TITS’2022 [76]   | 72.20   |
| IDF-TCSVT’2023 [84]      | 74.00   |
|--------------------------|---------|

**TABLE_3**:

| Ours    | 74.38   |
| Ours+   | 74.71   |
|---------|---------|


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_17 (`vis_17.jpg`)

- **Paper ID**: `AdWeatherNet_Adverse_Weather_Denoising_with_Point_Cloud_Spatiotemporal_Attention.pdf`
- **Page**: 2
- **Context Text** (前 500 字符): TABLE I
COMPARISON AMONG MAJOR POINT CLOUD DATASETS.


[TABLE_0_PLACEHOLDER]


A. Spatial Encoder
II. THE AdScenes DATASET
The existing encoding [17]–[19] often results in a notable
loss of features from distant point clouds, which are equally
important. To tackle this problem, we propose SE, a dynamic
encoder designed to assist the model in learning distinct
features at various distances. Assuming a uniform distribution
of multi-class targets in the outdoor scene, the density of point
clouds is

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Dataset           | Weather                                            |
|                   | Type Task Annotation                               |
|                   | Rain Snow Fog Multi-intensities                    |
|                   | √                                                  |
|:------------------|:---------------------------------------------------|
| nuScenes          | Real detection bbox × × ×                          |
| Waymo             | √                                                  |
| Oxford RobotCar   | Real detection bbox × × ×                          |
| CADC              | √ √                                                |
| STF               | Real detection bbox × ×                            |
|                   | √                                                  |
|                   | Real detection bbox × × ×                          |
|                   | √ √ √                                              |
|                   | Real detection bbox ×                              |
|                   | √                                                  |
| Panoptic nuScenes | Real segmentation point-wise × × ×                 |
| SemanticKITTI     | √ √                                                |
| Semantic STF      | Real segmentation point-wise × ×                   |
|                   | √ √ √                                              |
|                   | Real segmentation point-wise ×                     |
|                   | √                                                  |
| WADS              | Real denoising point-wise × × ×                    |
| SnowyKITTI        | √                                                  |
|                   | Synthetic denoising point-wise × × ×               |
|                   | √ √ √ √                                            |
| AdScenes          | Real/Synthetic denoising/detection point-wise/bbox |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_18 (`vis_18.jpg`)

- **Paper ID**: `Multi-Level_Feature_Fusion_Based_UAV_Object_Detection_Method_Under_Foggy_Weather.pdf`
- **Page**: 2
- **Context Text** (前 500 字符): improving the semantic information extraction for small 
targets[9]. Pan et al. introduced the concentration-based 
attention module in the YOLOv8’s backbone to enhance the 
feature extraction and semantic fusion ability[10]. Researchers 
also improve the loss function to increase the model’s detection 
accuracy for small targets. Wang et al. replaced the original 
intersection over union (IoU) loss function with wise-IoU v3, 
which improves the localization ability and reduces the leakage 
dete

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| FDM                       | Col1   | Col2   | Col3      |
| Conv                      |        |        |           |
| Conv                      |        |        |           |
| Fog Images                |        |        |           |
| C2f                       |        |        |           |
| Conv C2f                  |        |        |           |
| C2f MLM                   |        |        |           |
| Upsample Conv v10Detect   |        |        |           |
| SCDown                    |        |        |           |
| C2f MLM                   |        |        |           |
| C2f MLM C2f v10Detect     |        |        |           |
| SCDown Upsample           |        |        |           |
| C2fCIB SCDown             |        |        |           |
| SPPF MLM                  |        |        |           |
| PSA C2fCIB v10Detect      |        |        |           |
|:--------------------------|:-------|:-------|:----------|
|                           | C2fCIB |        | v10Detect |

**TABLE_1**:

| Col0   | CA PA         | Col2   |
|        | ... ... ...   |        |
|:-------|:--------------|:-------|

**TABLE_2**:

| Col0   | Col1   | Col2   | Col3   | Col4   | Col5   | Col6   | Col7   | Col8   |
|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|
|        |        |        |        |        |        |        |        |        |
| ...    |        | ...    |        |        | ...    |        |        |        |

**TABLE_3**:

| Col0   | CA attention PA attention                                                      | Col2   |
|        | Conv ReLU Conv Average Pooling Conv ReLU Conv Sigmoid Conv ReLU Conv Sigmoid   |        |
|:-------|:-------------------------------------------------------------------------------|:-------|

**TABLE_4**:

| 0-Average Pooling   | 1-Conv   | 2-ReLU   | 3-Conv   | 4-Sigmoid   | 5-Conv   | 6-ReLU Conv   | 7-Sigmoid   |
|---------------------|----------|----------|----------|-------------|----------|---------------|-------------|


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_19 (`vis_19.jpg`)

- **Paper ID**: `Weather-Aware_Drone-View_Object_Detection_Via_Environmental_Context_Understanding.pdf`
- **Page**: 4
- **Context Text** (前 500 字符): 

[TABLE_0_PLACEHOLDER]




[TABLE_1_PLACEHOLDER]


FFN
Bbox
Class
Transformer
Transformer
Image
Backbone
Encoder
Decoder


[TABLE_3_PLACEHOLDER]


Weather-aware Object Query 𝑸𝒘𝒐
Split into
Patches
⋮
Learnable Object Query 𝑸𝒐


[TABLE_4_PLACEHOLDER]


CLIP
Image Encoder
⋮
Selected Weather 
Content Query 𝑸𝒘
Select top K most similar weather 
content features for each image feature
Fig. 2. The overview of the proposed method. While the input aerial image is processed through a drone-view object de

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Col0   | Col1   | Col2   | Col3   | ⋮   |
|--------|--------|--------|--------|-----|

**TABLE_1**:

| Col0   | Col1   | Col2   | Col3   | Col4   | ⋮   | Col6   |
|--------|--------|--------|--------|--------|-----|--------|

**TABLE_2**:

| Col0   | Col1   | Col2   | Col3   | ⋮   |
|--------|--------|--------|--------|-----|

**TABLE_3**:

| Col0   | Col1   | Col2   | Col3   | Col4   | ⋮   | Col6   |
|--------|--------|--------|--------|--------|-----|--------|

**TABLE_4**:

| Col0   | W   | eath          | ⋮                     | nt    | Col5   |
|        | F   | Wea           | ⋮                     | ent   |        |
|        |     | eWaetu        | ⋮                     |       |        |
|        |     |               | ⋮                     |       |        |
|        |     |               | er Content            |       |        |
|        |     |               | ther Conte            |       |        |
|        |     |               | arteh eSre tC o𝒇n𝒑t   |       |        |
|:-------|:----|:--------------|:----------------------|:------|:-------|
|        |     | Feature Set 𝒇 |                       |       |        |
|        |     | 𝒑             |                       |       |        |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_20 (`vis_20.jpg`)

- **Paper ID**: `Nakamura_Active_Domain_Adaptation_with_False_Negative_Prediction_for_Object_Detection_CVPR_2024_paper.pdf`
- **Page**: 4
- **Context Text** (前 500 字符): 

[TABLE_0_PLACEHOLDER]


60
Algorithm 1: Training procedure of proposed method
Number of Errors
50
Input: Source data DS, Target data DT = {DLT , DUT },
where DLT = ∅
Output: Model parameters θt, θs adapted on target
domain
40
30
20
1 begin
10
2
Pre-train the student model θs, ϕs based on Eq. (1)
3
θt ←θs, ϕt ←ϕs
4
for R Rounds do
5
Freeze θt and train FNPM ψ based on Eq. (4)
Figure 3. Comparison of the number of false positive (FP) er-
rors and FN errors in unsupervised domain adaptation (UDA)

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Col0   | Col1   | Col2   |
|:-------|:-------|:-------|
|        |        |        |
|        |        |        |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_21 (`vis_21.jpg`)

- **Paper ID**: `Chen_Tokenize_Image_Patches_Global_Context_Fusion_for_Effective_Haze_Removal_CVPR_2025_paper.pdf`
- **Page**: 2
- **Context Text** (前 500 字符): ranging from 256 256 to 512 512 pixels. Constrained
by GPU memory, these methods often employ compromises
when processing large inputs, resorting to strategies such
as slicing and downsampling [15, 23, 37, 62]. Although
image slicing allows the processing of large inputs, it dis-
rupts global contextual information, potentially leading to a
loss in object coherence and spatial relationships. On the
other hand, downsampling preserves global structure but
sacriﬁces critical high-frequency details 

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Col0   | Col1   | Col2   |
|:-------|:-------|:-------|
| D      |        | XL     |
|        |        | )      |
|        | ehaze  | XL     |
|        | (ours  |        |

**TABLE_1**:

| Col0   |   Col1 | GB   |
|:-------|-------:|:-----|
|        |  20.04 |      |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_22 (`vis_22.jpg`)

- **Paper ID**: `Object Detection in Autonomous Vehicles under Adverse
Weather: A Review of Traditional and Deep
Learning Approaches.pdf`
- **Page**: 17
- **Context Text** (前 500 字符): Algorithms 2024, 17, 103
17 of 36


[TABLE_0_PLACEHOLDER]


Figure 12. Structure of YOLOv8.
The distinction between traditional and DL approaches is justified by their fundamen-
tally different methodologies and performance in various weather conditions. Traditional
methods, while effective in ideal conditions, struggle with adverse weather due to their
sensitivity to lighting, occlusion, and the need for precise feature engineering. DL methods,
on the other hand, have shown superior performance

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Conv       | Neck         | Head     |
| Conv       | C2f          | Detect   |
| C2f        | Conv         | Detect   |
| Conv       | Concat       | Detect   |
| C2f        | upsample     |          |
| Conv       | C2f Concat   |          |
| C2f        | Concat       |          |
| Conv       | C2f          |          |
| C2f        | Conv         |          |
| SPPF       | upsample     |          |
| Backbone   | Concat       |          |
|            | C2f          |          |
|:-----------|:-------------|:---------|


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_23 (`vis_23.jpg`)

- **Paper ID**: `Assessment_of_Efficient_and_Cost-Effective_Vehicle_Detection_in_Foggy_Weather.pdf`
- **Page**: 5
- **Context Text** (前 500 字符): 2024 18th International Conference on Open Source Systems and Technologies (ICOSST)
TABLE II
VEHICLE DETECTION RESULTS (ACCURACY) OF YOLO-V11 ON DAWN DATASET


[TABLE_0_PLACEHOLDER]


TABLE III
VEHICLE DETECTION RESULTS (SPEED) OF YOLO-V11 ON DAWN DATASET


[TABLE_1_PLACEHOLDER]


TABLE IV
VEHICLE DETECTION RESULTS (ACCURACY) OF YOLO-V11 ON FD DATASET


[TABLE_2_PLACEHOLDER]


TABLE V
VEHICLE DETECTION RESULTS (SPEED) OF YOLO-V11 ON FD DATASET


[TABLE_3_PLACEHOLDER]


and 71 % F1 score on the D

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Model     |   % mAP50 Car |   % mAP50 Bus |   % mAP50 Truck |   % mAP50 |   % mAP50-95 |   % F1 score |   % P |   % R |
|:----------|--------------:|--------------:|----------------:|----------:|-------------:|-------------:|------:|------:|
| YOLO-V11n |          85.6 |          31.9 |            69.7 |      59.7 |         36.1 |           55 |  63.7 |  59.3 |
| YOLO-V11s |          88.2 |          36.3 |            68.4 |      73.1 |         38.5 |           71 |  73.8 |  69.1 |
| YOLO-V11m |          89.1 |          33.2 |            69.8 |      72.9 |         38.3 |           71 |  75.7 |  69.5 |
| YOLO-V11l |          87.9 |          36.3 |            67.2 |      72.7 |         37.2 |           69 |  80   |  63.7 |

**TABLE_1**:

| Model     |   Pre-processing Time (ms) |   Inference Time (ms) |   Post-processing Time (ms) |   FPS (Inference Speed) |
|:----------|---------------------------:|----------------------:|----------------------------:|------------------------:|
| YOLO-V11n |                        0.4 |                  30.4 |                        18   |                      21 |
| YOLO-V11s |                        0.6 |                  26.5 |                        11.1 |                      26 |
| YOLO-V11m |                        0.3 |                  29.5 |                         9.4 |                      26 |
| YOLO-V11l |                        0.9 |                  42.4 |                        17.7 |                      16 |

**TABLE_2**:

| Model     |   % mAP50 Car |   % mAP50 Bus |   % mAP50 Truck |   % mAP50 |   % mAP50-95 |   % F1 score |   % P |   % R |
|:----------|--------------:|--------------:|----------------:|----------:|-------------:|-------------:|------:|------:|
| YOLO-V11n |          69.7 |          31.1 |            26.2 |      42.3 |         22.5 |           35 |  80.9 |  29.3 |
| YOLO-V11s |          67.4 |          41.9 |            17   |      42.1 |         21.5 |           37 |  71.1 |  37.8 |
| YOLO-V11m |          68.7 |          35.6 |            38.6 |      47.7 |         26.9 |           47 |  60.7 |  38.6 |
| YOLO-V11l |          66.4 |          41.2 |            29.3 |      45.6 |         24.7 |           45 |  70.9 |  37.1 |

**TABLE_3**:

| Model     |   Pre-processing Time (ms) |   Inference Time (ms) |   Post-processing Time (ms) |   FPS (Inference Speed) |
|:----------|---------------------------:|----------------------:|----------------------------:|------------------------:|
| YOLO-V11n |                        1.8 |                  34.6 |                        41.4 |                      13 |
| YOLO-V11s |                        1.8 |                  30.1 |                        55.1 |                      12 |
| YOLO-V11m |                        2.8 |                  39.8 |                        38.4 |                      12 |
| YOLO-V11l |                        2   |                  47   |                        38.6 |                      11 |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_24 (`vis_24.jpg`)

- **Paper ID**: `AL-YOLO_Accurate_and_Lightweight_Vehicle_and_Pedestrian_Detector_in_Foggy_Weather.pdf`
- **Page**: 5
- **Context Text** (前 500 字符): TABLE II
COMPARISON OF YOLO VARIANTS AND OUR MODEL. X DENOTES THE C3RFE MODULE.


[TABLE_0_PLACEHOLDER]


C. Experimental Results and Implementation Details
We trained YOLOv3, YOLOv5s, and our proposed model
on the Foggy-Cityscapes dataset for 200 epochs. Input images
were resized to a fixed dimension of 640 640, and training
was conducted with a batch size of 16 to balance computa-
tional efficiency and memory usage. The initial learning rate
(lr0) was set to 0.01 for stochastic gradient descen

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Model        | Average Precision   | Col2   |   mAP@0.5 | Parameters   | Inference Time   |
|:-------------|:--------------------|:-------|----------:|:-------------|:-----------------|
|              | Pedestrian          | Car    |           |              |                  |
| IA-YOLO [2]  | 26.7                | 37.3   |     32    | 8.765 M      | 35 ms            |
| R-YOLOv3 [3] | 35.8                | 58.0   |     46.9  | 8 M          | 1.5 ms           |
| YOLOv3       | 24.5                | 50.4   |     37.45 | 8.6 M        | 1.5 ms           |
| YOLOv5s      | 35.3                | 60.9   |     48.1  | 7 M          | 5.3 ms           |
| Ours         | 37.8                | 61.6   |     49.7  | 4.4 M        | ≈8.5 ms          |
| Ours + X     | 38.4                | 61.8   |     50.1  | 4.7 M        | 8.7 ms           |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_25 (`vis_25.jpg`)

- **Paper ID**: `AdWeatherNet_Adverse_Weather_Denoising_with_Point_Cloud_Spatiotemporal_Attention.pdf`
- **Page**: 4
- **Context Text** (前 500 字符): Fig. 3. Visual results of point cloud denoising algorithms. (a) raw point clouds, where rainy scenes are on the top and snowy scenes are on the bottom. (b)
Results of DROR filter. (c) Results of DSOR filter. (d) Results of LIOR filter. (e) Results of WeatherNet. (f) Results of 4DenoiseNet. (g) Results of ours. *:
no training required.
TABLE II
QUANTITATIVE EVALUATION. TRAINING AND INFERENCE ARE BOTH CONDUCTED ON NVIDIA GTX 3090 GPU.
*: NO TRAINING REQUIRED. BOLD FONT: THE STATE-OF-THE-ART RESULT

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Col0         | *: NO TRAINING REQUIRED. B   | BOLD FONT: THE STATE-OF-THE-AR   | RT RESULTS.          |
|:-------------|:-----------------------------|:---------------------------------|:---------------------|
| Weather      | Rain                         | Snow                             | Fog                  |
|              | Precision Recall IoU         | Precision Recall IoU             | Precision Recall IoU |
| DROR*        | 83.4 91.2 81.6               | 71.5 91.9 63.6                   | 76.1 89.9 60.9       |
| DSOR*        | 79.8 91.4 77.5               | 76.6 94.7 74.6                   | 79.5 93.3 72.4       |
| LIOR*        | 70.6 80.7 59.8               | 72.4 83.5 63.3                   | 70.6 80.1 60.7       |
| WeatherNet   | 88.4 87.2 86.4               | 87.6 97.8 85.0                   | 83.7 95.2 76.5       |
| 4DenoiseNet  | 96.5 92.3 88.3               | 91.5 98.7 90.3                   | 93.2 96.2 85.2       |
| AdWeatherNet | 99.3 98.5 97.8               | 97.6 98.5 96.2                   | 96.8 96.4 87.1       |

**TABLE_1**:

| Weather   | mAP                                 |
|           | Raw 4DenoiseNet AdWeatherNet Gain   |
|:----------|:------------------------------------|
| Rain      | 73.52 76.33 79.91 +3.58             |
| Snow      | 69.78 73.33 74.59 +1.26             |
| Fog       | 75.87 77.52 78.12 +0.60             |

**TABLE_2**:

| Col0         | TABLE IV ABLATION STUDY.   |
|:-------------|:---------------------------|
| Metrics      | Precision Recall IoU       |
| Baseline     | 93.4 98.1 91.8             |
| Baseline-SE  | 95.8 98.5 94.4             |
| Baseline-TDA | 96.2 98.6 94.9             |
| Full Model   | 97.6 98.5 96.2             |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_26 (`vis_26.jpg`)

- **Paper ID**: `Chen_Tokenize_Image_Patches_Global_Context_Fusion_for_Effective_Haze_Removal_CVPR_2025_paper.pdf`
- **Page**: 4
- **Context Text** (前 500 字符): 

[TABLE_0_PLACEHOLDER]




[TABLE_1_PLACEHOLDER]


ĊĊ
ĊĊ
ĊĊ
Output
Output
Image Stitching
Image Stitching
ĊĊ
ĊĊ
ĊĊ
Figure 3. Overall architecture of the proposed model. It begins by partitioning the hazy image into uniform-sized patches, which are then
encoded into tokens by the Encoder. The Bottleneck injects global information into each token, enhancing the contextual representation.
Subsequently, the Decoder reconstructs the tokens back into image patches, forming the ﬁnal output image. Nota

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Col0                          | Encoder                                                                           | Col2                  | Col3          | Col4   | Col5                  | Col6    | Col7        | Col8    | Col9        | Col10   | Token 1                       | Input Image Patches                 |
|                               | Block Block Block Block                                                           |                       |               |        |                       |         |             |         |             |         | Token 2                       | ĊĊ                                  |
|                               | Embedding                                                                         |                       |               |        |                       |         |             |         |             |         | Token 3                       | Batching                            |
|                               | Transformer Merging Transformer Merging Transformer Merging Transformer Merging   |                       |               |        |                       |         |             |         |             |         | Token 4                       | ĊĊ                                  |
|                               | ĊĊ ĊĊ ĊĊ Ċ Linear Patch Patch Patch Patch                                         |                       |               |        |                       |         |             |         |             |         | Token 5 Rearrange             | Batch 1 Mini-Batch 2 Mini-Batch n   |
|                               | Swin Swin Swin Swin                                                               |                       |               |        |                       |         |             |         |             |         | Token 6 Embedding             | Encoder                             |
|                               |                                                                                   |                       |               |        |                       |         |             |         |             |         | ĊĊ                            | Batch Aggregation                   |
|                               |                                                                                   |                       |               |        |                       |         |             |         |             |         | +                             | Encoding                            |
|                               |                                                                                   |                       |               |        |                       |         |             |         |             |         | Concatenate                   | Bottleneck                          |
|                               |                                                                                   |                       |               |        |                       |         |             |         |             |         | Ċ Mini- ĊĊ Position           | Decoding /                          |
|                               |                                                                                   |                       |               |        |                       |         |             |         |             |         | Token n-1                     | Batching                            |
|                               |                                                                                   |                       |               |        |                       |         |             |         |             |         | Token n Bottleneck            | Decoder                             |
|                               |                                                                                   |                       |               |        |                       |         |             |         |             |         | Global Attention Module       | ĊĊ                                  |
|                               |                                                                                   |                       |               |        |                       |         |             |         |             |         | RMSNorm Block                 | Batch 1 Mini-Batch 2 Mini-Batch n   |
|                               |                                                                                   |                       |               |        |                       |         |             |         |             |         | Hyper Attention Transformer   | Batch Aggregation                   |
|                               |                                                                                   |                       |               |        |                       |         |             |         |             |         | +                             | ĊĊ                                  |
|                               |                                                                                   |                       |               |        |                       |         |             |         |             |         | Divide                        | Output Image Patches                |
|                               |                                                                                   |                       |               |        |                       |         |             |         |             |         | RMSNorm                       |                                     |
|                               |                                                                                   |                       |               |        |                       |         |             |         |             |         | Forward Feed + LLM            |                                     |
|                               |                                                                                   |                       |               |        |                       |         |             |         |             |         | Rearrange                     |                                     |
|                               |                                                                                   |                       |               |        |                       |         |             |         |             |         | Ċ + Mini-                     |                                     |
|                               |                                                                                   |                       |               |        |                       |         |             |         |             |         | LLM Transformer Block         |                                     |
|                               |                                                                                   |                       |               |        |                       |         |             |         |             |         | ĊĊ                            |                                     |
|                               |                                                                                   |                       |               |        |                       |         |             |         |             |         | LLM Transformer Block         |                                     |
|:------------------------------|:----------------------------------------------------------------------------------|:----------------------|:--------------|:-------|:----------------------|:--------|:------------|:--------|:------------|:--------|:------------------------------|:------------------------------------|
|                               | Linear Embedding                                                                  | Swin Transformer Bloc | Patch Merging | ĊĊ     | Swin Transformer Bloc | Merging | Bloc        | Merging | Bloc        | Merging |                               |                                     |
|                               |                                                                                   |                       |               |        |                       | Ċ Patch | Transformer | Ċ Patch | Transformer | Ċ Patch |                               |                                     |
|                               |                                                                                   |                       |               |        |                       |         | Ċ           |         | Ċ           |         |                               |                                     |
|                               |                                                                                   |                       |               |        |                       |         | Swin        |         | Swin        |         |                               |                                     |
| ×2 ×2 ×6                      |                                                                                   |                       |               |        |                       |         | ×6          |         | ×2          |         |                               |                                     |
| Skip Connection               |                                                                                   |                       |               |        |                       |         |             |         | Block       |         |                               |                                     |
| Convolution Block Block Block |                                                                                   |                       |               |        |                       |         |             |         | xpanding    |         |                               |                                     |
| xpanding xpanding xpanding    |                                                                                   |                       |               |        |                       |         |             |         | former      |         |                               |                                     |
| former former former          |                                                                                   |                       |               |        |                       |         |             |         |             |         |                               |                                     |
|                               | Transposed Trans E Trans E Trans E Trans E                                        |                       |               |        |                       |         |             |         |             |         |                               |                                     |
|                               | ĊĊ ĊĊ ĊĊ Ċ Patch Patch Patch Patch                                                |                       |               |        |                       |         |             |         |             |         |                               |                                     |
|                               | Swin Swin Swin Swin                                                               |                       |               |        |                       |         |             |         |             |         |                               |                                     |
|                               | ×2 ×2 ×6 ×2                                                                       |                       |               |        |                       |         |             |         |             |         |                               |                                     |
|                               | Decoder                                                                           |                       |               |        |                       |         |             |         |             |         |                               |                                     |

**TABLE_1**:

| Col0   | Col1   | Col2   | Col3   | Image Partition   |
|--------|--------|--------|--------|-------------------|


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_27 (`vis_27.jpg`)

- **Paper ID**: `Multi-Level_Feature_Fusion_Based_UAV_Object_Detection_Method_Under_Foggy_Weather.pdf`
- **Page**: 4
- **Context Text** (前 500 字符): the depth value derived from the smoothed depth maps. 
Moreover, to assess the performance of object detection 
algorithms under varying fog densities, we prepared five 
datasets, each with a distinct level of fog density ranging from 
light to heavy, using the method mentioned above.
(4)
B. Compare with the SOTA Methods
where d and u take values in the closed interval 0 to 1 and 
depend on different regression samples. And we set d to 0 and 
u to 0.835 by experiments.
We compare our method with

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Detection     |   P(%)a |   R(%)b |   mAP0.5(%) |   mAP0.5:0.9(%) |
| algorithm     |         |         |             |                 |
|:--------------|--------:|--------:|------------:|----------------:|
| Faster R-CNN  |    42.1 |    24.7 |        22.8 |            16.3 |
| IA-YOLO       |    44.5 |    37.2 |        37.5 |            23.9 |
| Fog Guard     |    48.2 |    33.4 |        38.4 |            22.6 |
| DE-YOLO       |    45.7 |    26.9 |        36.9 |            22.4 |
| YOLOv10s      |    50.8 |    35.3 |        36.7 |            21.9 |
| DU-YOLO(ours) |    52   |    38.3 |        41.6 |            24.6 |

**TABLE_1**:

| Detection     |   P(%)a |   R(%)b |   mAP0.5(%) |   mAP0.5:0.9(%) |
| algorithm     |         |         |             |                 |
|:--------------|--------:|--------:|------------:|----------------:|
| Faster R-CNN  |    34.7 |    19.4 |        18.4 |            14.3 |
| IA-YOLO       |    44.9 |    25.2 |        26.1 |            16.1 |
| Fog Guard     |    46.1 |    25.3 |        26.7 |            15.9 |
| DE-YOLO       |    45.7 |    25.4 |        25.9 |            15.5 |
| YOLOv10s      |    45.3 |    24.6 |        25.6 |            15.4 |
| DU-YOLO(ours) |    46.3 |    26.1 |        27.1 |            16.5 |

**TABLE_2**:

| Detection     |   P(%)a |   R(%)b |   mAP0.5(%) |   mAP0.5:0.9(%) |
| algorithm     |         |         |             |                 |
|:--------------|--------:|--------:|------------:|----------------:|
| Faster R-CNN  |    30.2 |    14.8 |        16.8 |            10.3 |
| IA-YOLO       |    41.3 |    16   |        17.7 |            10.9 |
| Fog Guard     |    42   |    16.5 |        18.1 |            11   |
| DE-YOLO       |    41.7 |    16.1 |        17.9 |            10.6 |
| YOLOv10s      |    41.1 |    15.9 |        17.6 |            10.4 |
| DU-YOLO(ours) |    42.6 |    16.7 |        18.3 |            11.2 |

**TABLE_3**:

| Detection    |   P(%)a |   R(%)b |   mAP0.5(%) |   mAP0.5:0.9(%) |
| algorithm    |         |         |             |                 |
|:-------------|--------:|--------:|------------:|----------------:|
| Faster R-CNN |    28.4 |     8.7 |        11.9 |             7.2 |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_28 (`vis_28.jpg`)

- **Paper ID**: `Object Detection in Autonomous Vehicles under Adverse
Weather: A Review of Traditional and Deep
Learning Approaches.pdf`
- **Page**: 10
- **Context Text** (前 500 字符): Algorithms 2024, 17, 103
10 of 36


[TABLE_0_PLACEHOLDER]




[TABLE_1_PLACEHOLDER]




[TABLE_2_PLACEHOLDER]




[TABLE_3_PLACEHOLDER]




[TABLE_4_PLACEHOLDER]


Figure 6. Weather impacts on AVs.
3.5.1. Performance of Sensors
Adverse weather conditions on roads provide serious difficulties for sensors in AVs.
The current sensing technologies, including LiDAR and cameras, perform well in clear
weather but have trouble when the roadway is covered with snow and may contain
erroneous reflections c

### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Technical parameters   | Col1             |
|:-----------------------|:-----------------|
| General Weather        | Road Weather     |
| Slow down when         | Extended         |
| it's foggy outside.    | headways are     |
|                        | necessary in     |
|                        | foggy scenarios. |

**TABLE_1**:

| Sensor performance   | Col1           |
|:---------------------|:---------------|
| General Weather      | Road Weather   |
| The sensor           | Snow cover     |
| detection range      | lane markers   |
| might be affected    | have an impact |
| by persistent rain   | on LiDAR       |
| or snow.             | performance.   |

**TABLE_2**:

| Driver behavior    | Col1            |
|:-------------------|:----------------|
| General Weather    | Road Weather    |
| Decreased          | A longer        |
| visibility impacts | stopping        |
| the speed of       | distance is     |
| conventional       | required in icy |
| vehicle drivers.   | conditions.     |

**TABLE_3**:

| Communication   | Col1              |
|:----------------|:------------------|
| General Weather | Road Weather      |
| Thunderstorms   | Snow, slush, or   |
| may interfere   | road debris can   |
| with            | harm vehicle      |
| communication   | comm devices,     |
|                 | resulting in link |
|                 | disruption.       |

**TABLE_4**:

| Cameras performance   | Col1           |
|:----------------------|:---------------|
| General Weather       | Road Weather   |
| The cameras can't     | A detecting    |
| identify obstacles    | error might    |
| in dense snow or      | arise from ice |
| fog.                  | reflection.    |


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---

## Image vis_29 (`vis_29.jpg`)

- **Paper ID**: `Nakamura_Active_Domain_Adaptation_with_False_Negative_Prediction_for_Object_Detection_CVPR_2024_paper.pdf`
- **Page**: 1
- **Context Text** (前 500 字符): This CVPR paper is the Open Access version, provided by the Computer Vision Foundation.
Except for this watermark, it is identical to the accepted version;
the final published version of the proceedings is available on IEEE Xplore.
Active Domain Adaptation with False Negative Prediction
for Object Detection
Yuzuru Nakamura1
Yasunori Ishii1
Takayoshi Yamashita2
1Panasonic Holdings Corporation
2Chubu University
{nakamura.yuzuru,ishii.yasunori}@jp.panasonic.com
takayoshi@isc.chubu.ac.jp
Abstract




### 配套表格/图表数据 (Markdown)


**TABLE_0**:

| Un   | Col1   |
| Un   |        |
| (O   |        |
|:-----|:-------|


### 你的分析 (请填写)

> _(在此处填写分析结果)_

---
