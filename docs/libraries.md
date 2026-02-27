# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/EndoResearch/cai-capsule-endoscopy.git
cd cai-capsule-endoscopy
```

## 2. Requirements

**Python:** 3.9 or newer.

**PyTorch & torchvision:** Core dependencies for reproducing all results. The following configuration has been tested and verified:

| Package | Version |
|---|---|
| CUDA | ≥ 11.8 |
| PyTorch | ≥ 2.3.1+cu121 |
| torchvision | ≥ 0.18.1 |

**PyTorch Lightning:** Used as the deep learning training framework. Tested with version `2.2.5`. To verify your installed version, run:

```python
import pytorch_lightning as pl
print(pl.__version__)  # Expected: 2.2.5
```
