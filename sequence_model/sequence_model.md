# Training: Single Task Classification
## 1. **Run training multi-frame classification:** 
Open a terminal and navigate to the directory containing `STL-multi-frame-training.py` and define the parameters:

### Anatomy classification
```bash
python STL-multi-frame-training.py \
  --feature_path ../features/Galar/GastroNet \
  --output_dir ../image_classification/STL-multi-frame/anatomy \
  --dataframe ../data/Galar/galar-labels_anatomy_anomaly_set_type.csv \
  --data_path ../data/Galar \
  --label anatomy \
  --batch_size 256 \
  --num_workers 0
```  
### Anomaly classification
```bash
python STL-multi-frame-training.py \
  --feature_path ../features/Galar/GastroNet \
  --output_dir ../image_classification/STL-multi-frame/anatomy \
  --dataframe ../data/Galar/galar-labels_anatomy_anomaly_set_type.csv \
  --data_path ../data/Galar \
  --label anomaly \
  --batch_size 256 \
  --num_workers 0
```  

# Training: Multi Task Classification
## 1. **Run training multi-frame classification:** 
Open a terminal and navigate to the directory containing `MTL-multi-frame-training.py` and define the parameters:

```bash
python MTL-multi-frame-training.py \
  --feature_path ../features/Galar/GastroNet \
  --output_dir ../image_classification/MLT-multi-frame/anatomy \
  --dataframe ../data/Galar/galar-labels_anatomy_anomaly_set_type.csv \
  --data_path ../data/Galar \
  --batch_size 256 \
  --num_workers 0
```  
- a short “Expected outputs” section (what files are produced in `--output_dir`)
- a “Common errors” section (missing paths, mismatch between dataframe filenames and `data_path`, etc.)