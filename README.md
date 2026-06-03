# DG-based IFD: DIS
Code for "Domain interference suppression for reliable fault diagnosis under unseen operating conditions". <br>
This repository provides PyTorch code for DIS training on the SQV bearing dataset.

## Run
```bash
pip install -r requirements.txt
python train.py --dataset SQV
```

For another dataset, i.e., CWRU:
```bash
python train.py --dataset CWRU --data_dir ./data/CWRU/
```

### SQV Bearing Dataset

The SQV bearing dataset can be downloaded from:
- Google Drive: https://drive.google.com/file/d/1ld6Qv95zYdql0HhEvET5tA-44gLA-G1_/view
- Dataset description (Chinese): https://blog.csdn.net/weixin_43543177/article/details/121549538

After downloading, prepare the data using:

## Data Format
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

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{2026-114457,
  title   = {Domain interference suppression for reliable fault diagnosis under unseen operating conditions},
  volume = {256},
  pages = {114457},
  year = {2026},
  doi = {https://doi.org/10.1016/j.ymssp.2026.114457},
}
```
