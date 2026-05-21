# DG-based IFD: DIS
This repository provides PyTorch code for DIS training on SQV bearing dataset.

## Run
```bash
pip install -r requirements.txt
python train.py --dataset SQV
```

For CWRU:
```bash
python train.py --dataset CWRU --data_dir ./data/CWRU/
```

## Data Format

Data files are not included in the repository.

![SQV data preparation and file structure](assets/data_preparation.svg)

Prepare SQV files from the original `.mat` files:

```bash
python scripts/prepare_sqv.py --mat_dir /path/to/SQV-public/mat_file --output_dir ./data/SQV/
```

Expected SQV files:
```text
data/SQV/SQV_x.npy
data/SQV/SQV_y.npy
```

The label array should contain two columns:
```text
class_label, domain_label
```
