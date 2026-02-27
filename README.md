# Computer-Assisted Intervention in Capsule Endoscopy: A Real-Time Edge-AI Auditing System
<img src="figures/fig1.jpg" alt="Pipeline" width="700">

Welcome to the anonymous repository for our MICCAI 2026 paper, currently under double-blind peer review. Here, you'll find scripts, datasets, and models essential and software on edge AI device for our research. 🚀

## Repository Structure
- image_classification/
- sequence_model/
- edge_deployment/
- docs/
- notebooks/

## Datasets
📊 Data
Summary: 

🔗 **(2025) Galar Dataset:** [Figshare](https://plus.figshare.com/articles/dataset/Galar_-_a_large_multi-label_video_capsule_endoscopy_dataset/25304616/1)

🔗 **Rhode island Dataset:** [Figshare](https://springernature.figshare.com/collections/Rhode_island_gastroenterology_video_capsule_endoscopy_data_set/6071216/1)

Note: This paper use uses open-access data.

This section provides an overview of the datasets used in our study 📌.
- 📼 Video Capsule Endoscopy-VCE for anatomical classification and anonaly classification: 80 patients (∼3.5 million frames).

📂 For more details: Check out the [data.md](data/data.md) file for a comprehensive guide on data organization and preprocessing steps.


## 🎯 Multi-Frame Embedding

Embedding Methods:
- 2️⃣ ResNet50 Pretrained on ImageNet
- 3️⃣ ResNet50 Pretrained on Endoscopy

📂 Learn More: Check the [features.md](features/features.md) file for detailed embedding representations of each video capsule endoscopy feature extraction.

## 🏷️ Task 1/2 Anatomical Classification: Single-frame and Multi-frame

Summary of Experiments


### Performance comparison between single-frame and multi-frame approaches for anatomical classification

Models were trained/validated in 60 studies (~3.2M frames) and evaluated in 20 independent test studies (~234K frames). 95% confidence intervals were calculated using stratified bootstrapping with 1,000 resamples by case. Results are reported as class-wise F1-scores and macro F1. HMM denotes Hidden Markov Model post-processing and the column reports the resulting macro F1-score. STL: Single-Task Learning (task: anatomy) and MTL: Multi-Task Learning (tasks: anatomy, anomaly). **\*** indicates deployment on the Edge AI device.

---

**Ablation Study F1-score (%) — Anatomical Task — Single-frame**

| Token = 1 | Mouth | Esophag. | Stomach | S. Bowel | Colon | Macro F1 | HMM | Download |
|---|---|---|---|---|---|---|---|---|
| ImageNet | 10.1±5.0 | 26.7±15.6 | 61.2±11.5 | 90.0±3.6 | 68.5±12.1 | 51.3±6.0 | 88.7±6.5 | [Download](https://drive.google.com/uc?id=XXX) |
| GastroNet | 60.9±11.3 | 69.0±18.4 | 82.4±7.5 | 94.0±3.5 | 81.0±10.4 | 77.5±5.8 | 93.3±4.4 | [Download](https://drive.google.com/uc?id=XXX) |

**Tokens ≥ 5 — Endoscopy-pretrained CNN+ViT (Multi-frame)**

| | Mouth | Esophag. | Stomach | S. Bowel | Colon | Macro F1 | HMM | Download |
|---|---|---|---|---|---|---|---|---|
| STL 5 | 80.7±8.7 | 87.2±9.9 | 89.6±4.6 | 95.3±3.4 | 85.6±11.1 | 87.7±5.0 | 93.7±4.9 | [Download](https://drive.google.com/uc?id=XXX) |
| STL 31 | 88.8±6.4 | 91.1±7.1 | 93.7±3.5 | 94.5±3.6 | 81.8±12.0 | 90.0±4.6 | 93.9±3.9 | [Download](https://drive.google.com/uc?id=XXX) |
| STL 75 | 88.4±6.5 | 89.8±6.6 | 93.3±4.3 | 95.4±3.7 | 84.6±11.7 | 90.3±5.0 | 93.9±3.9 | [Download](https://drive.google.com/uc?id=XXX) |
| STL 195 | 87.4±7.2 | 89.3±7.8 | 93.3±3.7 | 96.0±3.6 | 87.3±12.0 | 90.7±4.7 | 93.2±4.2 | [Download](https://drive.google.com/uc?id=XXX) |
| MTL 5 | 58.7±20.2 | 72.8±17.5 | 85.6±6.5 | 94.5±3.5 | 82.8±10.6 | 78.9±6.5 | 92.4±5.1 | [Download](https://drive.google.com/uc?id=XXX) |
| MTL 31 | 70.0±19.1 | 89.7±7.7 | 93.1±3.5 | 94.4±3.8 | 81.7±11.8 | 85.8±6.5 | 93.8±4.0 | [Download](https://drive.google.com/uc?id=XXX) |
| **MTL 75\*** | 87.0±7.6\* | 89.5±7.7\* | 93.0±4.2\* | 95.9±3.5\* | 86.9±11.2\* | 90.3±5.3\* | 93.0±4.2 | [Download](https://drive.google.com/uc?id=XXX) |
| MTL 195 | 22.9±12.1 | 67.4±18.2 | 92.5±4.4 | 95.4±3.5 | 85.5±11.2 | 72.8±6.4 | 92.4±4.1 | [Download](https://drive.google.com/uc?id=XXX) |

---

**Comparison with state-of-the-art single-frame approaches (same dataset)**

| Token = 1 | Mouth | Esophag. | Stomach | S. Bowel | Colon | Macro F1 | HMM | Link Paper |
|---|---|---|---|---|---|---|---|---|
| Le et al. 2025 | 42.0 | 65.0 | 78.0 | 93.0 | 75.0 | 71.0 | - | [Figshare](https://www.nature.com/articles/s41597-025-05112-7) |
| **Werner et al. 2025** | - | - | - | - | - | 65.1 | 92.4 | [Figshare](https://arxiv.org/pdf/2507.23479) |

<!--
📂 For a detailed breakdown, refer to the [image_classification.md](image_classification/image_classification.md), [sequence_model.md](sequence_model/image_classification.md)  file.
-->
📂 The trained models are available. However, the training scripts for anatomical classification will be available after the peer-review process is completed.


## 🏥 Task 2/2 Anomaly Classification: Single-frame and Multi-frame

<!--
📂 For a detailed breakdown, refer to the [image_classification.md](image_classification/image_classification.md), [sequence_model.md](sequence_model/image_classification.md)  file.
-->
📂 The trained models are available. However, the training scripts will be available after the peer-review process is completed.

### Performance comparison of single-frame and multi-frame approaches for binary anomaly detection

Same data split and 95% CI estimation as in Table 1. **\*** Deployment on the edge AI device. **Note:** Anomaly category was extended to include inflammatory bowel disease, foreign body, hematin, cancer, and lymphangiectasis, as these findings are clinically considered abnormal.

---

**Task 2/2 Anomaly — Single-frame and Multi-frame (STL, MTL)**

| Method | Micro F1 | Macro F1 | Macro Prec. | Macro Rec. | Model | Embedded |
|---|---|---|---|---|---|---|
| **Werner et al. 2025** | 87.7 | 54.4 | 54.1 | 54.7 | ResNet50 | No |
| Werner et al. 2025 | 87.3 | 37.0 | 26.6 | 60.9 | MobileNet-S | No |
| GastroNet | 79.5±6.9 | 61.6±8.4 | 61.7±8.4 | 62.4±9.1 | ResNet50 | Yes |
| STL 75 | 85.7±9.3 | 59.7±11.1 | 76.8±15.7 | 58.2±7.8 | CNN+ViT | Yes |
| **MTL 75\*** | 77.0±6.2\* | 62.8±10.3\* | 61.9±9.4\* | 66.7±13.8\* | CNN+ViT | Yes |


## 📊 Report Quality Indicators


<img src="figures/fig2.jpg" alt="QI" width="700">

## 🔨 Installation
Please refer to the [libraries.md](docs/libraries.md) file for detailed installation instructions.



## 📓 Notebooks
`predict_example.ipynb`:  Use this notebook to run sequence classification tasks for inference.

Note 🗈:  To run this code in Google Colab, click the logo:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/XXX)



