# STANet for remote sensing image change detection

It is the implementation of the paper: A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection.

Here, we provide the pytorch implementation of the spatial-temporal attention neural network (STANet) for remote sensing image change detection.

![image-20200601213320103](/src/stanet-overview.png)

## Prerequisites

- windows or Linux 
- Python 3.6+
- CPU or NVIDIA GPU
- CUDA 9.0+
- PyTorch > 1.0
- visdom

## Installation

Clone this repo:

```bash
git clone https://github.com/justchenhao/STANet
cd STANet
```

Install [PyTorch](http://pytorch.org/) 1.0+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate))

## Prepare Datasets

### download the change detection dataset

You could download the LEVIR-CD at https://justchenhao.github.io/LEVIR/;

The path list in the downloaded folder is as follows:

```
path to LEVIR-CD:
                ├─train
                │  ├─A
                │  ├─B
                │  ├─label
                ├─val
                │  ├─A
                │  ├─B
                │  ├─label
                ├─test
                │  ├─A
                │  ├─B
                │  ├─label
```

where A contains images of pre-phase, B contains images of post-phase, and label contains label maps.

### cut bitemporal image pairs

The original image in LEVIR-CD has a size of 1024 * 1024, which will consume too much memory when training. Therefore, we can cut the origin images into smaller patches (e.g., 256 * 256, or 512 * 512).  In our paper, we cut the original image into patches of 256 * 256 size without overlapping.

Make sure that the corresponding patch samples in the A, B, and label subfolders have the same name.

## Train

### Monitor training status

To view training results and loss plots, run this script and click the URL [http://localhost:8097](http://localhost:8097/).

```bash
python -m visdom.server
```

### train with our base method

Run the following script:

```bash
python ./train.py --save_epoch_freq 1 --angle 15 --dataroot path-to-LEVIR-CD-train --val_dataroot path-to-LEVIR-CD-val --name LEVIR-CDF0 --lr 0.001 --model CDF0 --batch_size 8 --load_size 256 --crop_size 256 --preprocess rotate_and_crop
```

Once finished, you could find the best model and the log files in the project folder.

### train with Basic spatial-temporal Attention Module (BAM) method

```bash
python ./train.py --save_epoch_freq 1 --angle 15 --dataroot path-to-LEVIR-CD-train --val_dataroot path-to-LEVIR-CD-val --name LEVIR-CDFA0 --lr 0.001 --model CDFA --SA_mode BAM --batch_size 8 --load_size 256 --crop_size 256 --preprocess rotate_and_crop
```

### train with Pyramid spatial-temporal Attention Module (PAM) method

```bash
python ./train.py --save_epoch_freq 1 --angle 15 --dataroot path-to-LEVIR-CD-train --val_dataroot path-to-LEVIR-CD-val --name LEVIR-CDFAp0 --lr 0.001 --model --SA_mode PAM CDFA --batch_size 8 --load_size 256 --crop_size 256 --preprocess rotate_and_crop
```

## Test

You could edit the file val.py, for example:

```python
if __name__ == '__main__':
    opt = TestOptions().parse()   # get training options
    opt = make_val_opt(opt)
    opt.phase = 'val'
    opt.dataroot = 'path-to-LEVIR-CD-test' # data root 
    opt.dataset_mode = 'changedetection'
    opt.n_class = 2
    opt.SA_mode = 'PAM' # BAM | PAM 
    opt.arch = 'mynet3'
    opt.model = 'CDFA' # model type
    opt.name = 'LEVIR-CDFAp0' # project name
    opt.results_dir = './results/' # save predicted images 
    opt.epoch = 'best-epoch-in-val' # which epoch to test
    opt.num_test = np.inf
    val(opt)
```

then run the script: `python val.py`. Once finished, you can find the prediction log file in the project directory and predicted image files in the result directory.

## Pre-train Module

Here, we provide the PAM pre-train model: [baidu-download-link, code:l3o7](https://pan.baidu.com/s/1ysWk-ljJ0wh_YEURiC_wsw), [google-download-link](https://drive.google.com/drive/folders/1QZOYP-N2Nr27aDG38zTVJMunfzmrseLv?usp=sharing).

## Other TIPS

For more Training/Testing guides, you could see the option files in the  `./options/`  folder.

## Citation

If you use this code for your research, please cite our papers.

```
@Article{chen2020,
AUTHOR = {Chen, Hao and Shi, Zhenwei},
TITLE = {A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection},
JOURNAL = {Remote Sensing},
VOLUME = {12},
YEAR = {2020},
NUMBER = {10},
ARTICLE-NUMBER = {1662},
URL = {https://www.mdpi.com/2072-4292/12/10/1662},
ISSN = {2072-4292},
DOI = {10.3390/rs12101662}
}
```

## Acknowledgments

Our code is inspired by [pytorch-CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

 