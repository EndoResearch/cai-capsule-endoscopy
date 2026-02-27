# 📂 Data Preparation Galar Dataset

## 1️⃣ Download the Galar Dataset

The proposed framework was evaluated using the **Galar – A Large Multi-Label Video Capsule Endoscopy Dataset**.

🔗 Available on Figshare:  
https://plus.figshare.com/articles/dataset/Galar_-_a_large_multi-label_video_capsule_endoscopy_dataset/25304616

---

## 2️⃣ Extract Dataset Files

Unzip the downloaded archives into the following directory: `data/Galar`.

📁 The directory should contain:
```
- 1
- 2
- 3
- 4
- ...
- 80
- Galar_labels_and_metadata
- Galar_splits
```

## 3️⃣ Download Additional Metadata
Download the provided `metadata.csv` file and place it inside the `data/Galar` directory: 
[Download metadata.csv](https://drive.google.com/uc?id=1sNvOBwYKYyJ8b4Auu66FbDBEAtPKgJYZ) 

```python
#validate data

import pandas as pd

df = pd.read_csv("galar-labels_anatomy_anomaly_set_type.csv",index_col=0)
df.head(3) 

path	anatomy	anomaly	set_type
0	1/frame_000100.PNG	mouth	0	train
1	1/frame_000105.PNG	mouth	0	train
2	1/frame_000110.PNG	mouth	0	train
```


# 📂 Data Preparation Rhode island Datase
## 1️⃣ Download the Rhode Island Dataset
For cross-dataset robustness evaluation, we used the **Rhode Island Gastroenterology Video Capsule Endoscopy Dataset**.

🔗 Available on Figshare:  
https://springernature.figshare.com/collections/Rhode_island_gastroenterology_video_capsule_endoscopy_data_set/6071216/1


## 2️⃣ Extract Dataset Files
Unzip the downloaded archives into the following directory: `data/Rhode_island`.

📁 The directory should contain:
```
- s001
- s010
- s016
- s018
- ...
- s424
```
Note: Remember that for robutness prub we use the test set (85 cases)
## 3️⃣ Download Additional Metadata
Download the provided `metadata.csv` file and place it inside the `data/Galar` directory: 
[Download metadata.csv](https://drive.google.com/uc?id=1e-nKIxinF0AW-yEfNhxxwK-ehRhBvORO) 

```python
#validate data

import pandas as pd

df = pd.read_csv("RhodeIsland_VCE_test_data.csv",index_col=0)
df.head(3)


patient	filename	section	filepath	frame	set_type
0	s001	image-00026.png	1_esophagus	RhodeIsland_VCE\s001\image-00026.png	26	test
1	s001	image-00027.png	1_esophagus	RhodeIsland_VCE\s001\image-00027.png	27	test
2	s001	image-00028.png	1_esophagus	RhodeIsland_VCE\s001\image-00028.png	28	test
```
