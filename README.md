# MaskSup
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/hasibzunair/masksup-segmentation-demo)

This is official code for our BMVC 2022 Oral paper:<br>
[Masked Supervised Learning for Semantic Segmentation](https://arxiv.org/abs/2210.00923)
<br>

![attention](https://github.com/hasibzunair/masksup-segmentation/blob/master/media/pipeline.png)

## 1. Specification of dependencies

This code requires Python 3.8.12. Run the following to install the required packages.
```
conda update conda
conda env create -f environment.yml
conda activate msl 
```

## 2a. Get datasets

First, open a folder named 
`datasets` in the root folder (`mkdir datasets`). Then, download GLaS, Kvasir & CVC-ClinicDB and NYUDv2 datasets as well as the sribbles from [GitHub Releases](https://github.com/hasibzunair/masksup-segmentation/releases/tag/v1.0). Finally, unzip and move the four folder to `datasets`.


## 2b. Train & Evaluation code
To train and evaluate MaskSup on GLaS or Kvasir & CVC-ClinicDB datasets, you need to change the `EXPERIMENT_NAME` in `trainval_glas_polyp.py` to a name that has glas or polyp. For example to train on GLaS, set `EXPERIMENT_NAME = "glas_masksup"`. Then run:
```
python trainval_glas_polyp.py
```
To train and evaluate MaskSup on NYUDv2 dataset, run:
```
python trainval_nyudv2.py
```

All experiments are conducted on a single NVIDIA 3080Ti GPU. For additional implementation details and results, please refer to the supplementary material [here](https://github.com/hasibzunair/masksup-segmentation/blob/master/media/supplementary_materials.pdf).

## 3. Pre-trained models

We provide pretrained models on [GitHub Releases](https://github.com/hasibzunair/masksup-segmentation/releases/tag/v0.1) for reproducibility. 
|Dataset      | Backbone  |   mIoU(%)  |   Download   |
|  ---------- | -------   |  ------ |  --------   |
| GLaS     |LeViT-UNet 384  |  76.06  | [download](https://github.com/hasibzunair/masksup-segmentation/releases/download/v0.1/masksupglas76.06iou.pth)   |
| Kvasir & CVC-ClinicDB     |LeViT-UNet 384 | 84.02  | [download](https://github.com/hasibzunair/masksup-segmentation/releases/download/v0.1/masksuppolyp84.02iou.pth)  |
| NYUDv2        |U-Net++ |  39.31  |  [download](https://github.com/hasibzunair/masksup-segmentation/releases/download/v0.1/masksupnyu39.31iou.pth)   |


## 4. Demo
A HuggingFace Spaces demo of the model trained with MaskSup on NYUDv2 is available at https://huggingface.co/spaces/hasibzunair/masksup-segmentation-demo.

## 5. Citation

```bibtex
 @inproceedings{zunair2022masked,
    title={Masked Supervised Learning for Semantic Segmentation},
    author={Zunair, Hasib and Hamza, A Ben},
    booktitle={Proc. British Machine Vision Conference},
    year={2022}
  }
```

### Acknowledgements
This code base is built on top of the following repositories: 
* https://github.com/apple1986/LeViT-UNet
* https://github.com/France1/unet-multiclass-pytorch
* https://github.com/milesial/Pytorch-UNet



