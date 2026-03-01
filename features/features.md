

# ResNet 50 
1. **Pretrained on ImageNet** Run script open a terminal and navigate to the directory containing `features_ImageNet.py` and define the parameters:

    ```python
    python features_ImageNet.py \
    --output_dir ../features/Galar/ImageNet \
    --dataframe ../data/Galar/galar-labels_anatomy_anomaly_set_type.csv \
    --data_path ../data/Galar \
    --labels_path ../data/Galar/Galar_labels_and_metadata/Labels \
    --batch_size 256 \
    --num_workers 0
    ```
2. **Pretrained on GastroNet*:** Run script open a terminal and navigate to the directory containing `features_GastroNet.py` and define the parameters:

    ```bash
    python features_GastroNet.py \
    --weights_path ../RN50_Pretrained_Weights.pth \
    --output_dir ../features/Galar/GastroNet\
    --dataframe ../data/Galar/galar-labels_anatomy_anomaly_set_type.csv \
    --data_path ../data/Galar \
    --batch_size 256 \
    --num_workers 0

### Note:
1. Check the existence of the paths of: 
   - data_path
   - output_dir
   - dataframe 


For all samples the data normalization was performed using the mean and standard deviation.
  ```bash
# Define transform
import torch
from torchvision import transforms as T

transform = T.Compose([
    T.Resize((224,224), interpolation=Image.LANCZOS), 
    T.ToTensor(), 
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) 
  ```
