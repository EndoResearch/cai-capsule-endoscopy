# Training: Single-Frame Classification

## 1. Run training for single-frame classification

Open a terminal, navigate to the directory containing `single-frame-linear-probing.py`, and run one of the following commands depending on the target label.

### Anatomy classification

```bash
python single-frame-linear-probing.py \
  --feature_path ../features/Galar/GastroNet \
  --output_dir ../image_classification/single-frame/anatomy \
  --dataframe ../data/Galar/galar-labels_anatomy_anomaly_set_type.csv \
  --data_path ../data/Galar \
  --label anatomy \
  --batch_size 256 \
  --num_workers 0
```
### Anomaly classification 

```bash
python single-frame-linear-probing.py \
--feature_path ../features/Galar/GastroNet \
--output_dir ../image_classification/single-frame/anomaly \
--dataframe ../data/Galar/galar-labels_anatomy_anomaly_set_type.csv \
--data_path ../data/Galar \
--label anomaly \
--batch_size 256 \
--num_workers 0
   ```

- a short “Expected outputs” section (what files are produced in `--output_dir`)
- a “Common errors” section (missing paths, mismatch between dataframe filenames and `data_path`, etc.)