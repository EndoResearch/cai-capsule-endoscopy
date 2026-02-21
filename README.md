# Computer-Assisted Intervention in Capsule Endoscopy: A Real-Time Edge-AI Auditing System
<img src="figures/fig1.jpg" alt="Pipeline" width="700">

Welcome to the anonymous repository for our MICCAI 2026 paper, currently under double-blind peer review. Here, you'll find scripts, datasets, and models essential and software on edge AI device for our research. 🚀

📊 Data
Summary: 
🔗 **(2025) Galar Dataset:** [Figshare](https://plus.figshare.com/articles/dataset/Galar_-_a_large_multi-label_video_capsule_endoscopy_dataset/25304616/1)
🔗 **Rhode island Dataset:** [Figshare](https://springernature.figshare.com/collections/Rhode_island_gastroenterology_video_capsule_endoscopy_data_set/6071216/1)
🔗 **Code:** [GitHub](https://github.com/Endo2026/EndoReview.git)

This section provides an overview of the datasets used in our study 📌.
- 📼 Video Capsule Endoscopy-VCE for anatomical classification: 80 patients (∼3.5 million frames).

📂 For more details: Check out the [data.md](data.md) file for a comprehensive guide on data organization and preprocessing steps.

## 🎯 Multi-Frame Embedding

Embedding Methods:
- 2️⃣ ResNet50 Pretrained on ImageNet
- 3️⃣ ResNet50 Pretrained on Endoscopy

📂 Learn More: Check the [features.md](features.md) file for detailed embedding representations of each videoendoscopy and sequence feature extraction.

## 🏷️ Organ Classification

Summary of Experiments

🔍 Spatial-Based Classification
- 1️⃣ ResNet50 Pretrained on ImageNet + MLP
- 2️⃣ ResNet50 Pretrained on Endoscopy + MLP

    | Embedding            | Resolution | Precision | Recall | F1    | MCC   |Download                                                                     |
    |:------------------:  |:----------:|:---------:|:------:|:-----:|:-----:|:---------------------------------------------------------------------------:|
    | ConvNeXt (ImageNet)  | 1 frame    | XX.XX     | XX.XX  | XX.XX | XX.XX | [Download](https://drive.google.com/uc?id=1A0h6V5HLpqyoaMzdrLFH32ksbJr6J9VF)|
    | ConvNeXt (Endoscopy) | 1 frame    | XX.XX     | XX.XX  | XX.XX | XX.XX | [Download](https://drive.google.com/uc?id=1vVVVwEFlAPBLpiIjoQ5eJtY5rYg8fbH4)|

🔄 Multi-Frame-Based Classification

Summary of Experiments

⏳ Temporal-Based Classification with Attention Mechanisms

- 1️⃣ ViT-Small initialized with Random Weights
- 2️⃣ ViT-Small initialized with Endoscopy Pretraining

<!--
📂 For more details, refer to the [organclassification.md](organclassification.md) file.
-->
📂 The trained models are available. However, the training scripts and labels for organ classification will be available after the peer-review process is completed.

- 2️⃣ ViT-Small initialized with Random Weights


- 3️⃣ ViT-Small initialized with Endoscopy Pretraining

## 🏥 Anomaly Classification

**Summary of Experiments**
- 🔬 **Selected Embedding:** ResNet50 Pretrained on Endoscopy
- ⏳ **Temporal-Based Evaluation** using different time intervals:
  - 1️⃣ **ViT-Small initialized with Endoscopy Pretraining – [5- 195] tokens**
<!--
📂 For a detailed breakdown, refer to the [stomachsiteclassification.md](stomachsiteclassification.md) file.
-->
📂 The trained models are available. However, the training scripts will be available after the peer-review process is completed.

- 1️⃣ **ViT-Small initialized with Endoscopy Pretraining – [5- 195] tokens**


## 📊 Report Quality Indicators


<img src="figures/fig2.jpg" alt="QI" width="700">

## 🔨 Installation
Please refer to the [libraries.md](libraries.md) file for detailed installation instructions.



## 📓 Notebooks
`predict_example.ipynb`:  Use this notebook to run sequence classification tasks for inference.

Note 🗈:  To run this code in Google Colab, click the logo:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/XXX)



