# MaskSup

This is official code for our BMVC 2022 paper:<br>
[Masked Supervised Learning for SemanticSegmentation](https://arxiv.org/abs/2210.00923)
<br>

![attention](https://github.com/hasibzunair/masksup-segmentation/blob/master/utils/pipeline.png)

## 1. Specification of dependencies

This code requires Python 3.8.12. Run the following to install the required packages.
```
conda update conda
conda env create -f environment.yml
conda activate msl 
```

## 2a. Get datasets

First, open a folder named 
`datasets` in the root folder. Then, download GLaS, Kvasir & CVC-ClinicDB and NYUDv2 datasets as well as the sribbles from [GitHub Releases](https://github.com/hasibzunair/masksup-segmentation/releases/tag/v1.0).


## 2b. Train & Evaluation code
To train and evaluate MaskSup on GLaS or Kvasir & CVC-ClinicDB datasets, run:

```
python trainval_glas_polyp.py
```
To train and evaluate MaskSup on NYUDv2 dataset, run:

```
python trainval_nyudv2.py
```

## 3. Pre-trained models
## Validation
We provide pretrained models on [GitHub Releases](https://github.com/hasibzunair/masksup-segmentation/releases/tag/v0.1) for reproducibility. 


|Dataset      | Backbone  |   Head nums   |   mIou(%)  |  Resolution     | Download   |
|  ---------- | -------   |  :--------:   | ------ |  :---:          | --------   |
| GLaS     |LeViT-UNet 384  |     1         |  76.06  |  448x448 |[download](https://github.com/hasibzunair/masksup-segmentation/releases/download/v0.1/masksupglas76.06iou.pth)   |
| Kvasir & CVC-ClinicDB     |LeViT-UNet 384 |     1         |  84.02  |  448x448 |[download](https://github.com/hasibzunair/masksup-segmentation/releases/download/v0.1/masksuppolyp84.02iou.pth)  |
| NYUDv2        |U-Net++ |     4         |  39.31  |  448x448 |[download](https://github.com/hasibzunair/masksup-segmentation/releases/download/v0.1/masksupnyu39.31iou.pth)   |


## 4. Demo
TODO

## 5. Citation
TODO

### Acknowledgements
* https://github.com/apple1986/LeViT-UNet
* https://github.com/France1/unet-multiclass-pytorch
* https://github.com/milesial/Pytorch-UNet


