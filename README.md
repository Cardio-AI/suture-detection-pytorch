# Suture detection PyTorch

This repo contains the reference implementation of suture detection model in PyTorch for the paper
> **Point detection through multi-instance deep heatmap regression for sutures in endoscopy**
>
> Lalith Sharan, Gabriele Romano, Julian Brand, Halvar Kelm, Matthias Karck, Raffaele De Simone, Sandy Engelhardt  
>
> [Accepted, IJCARS 2021](https://doi.org/10.1007/s11548-021-02523-w)

Please see the [license file](LICENSE) for terms os use of this repo.
If you find our work useful in your research please consider citing our paper:

```
@article{sharan_point_2021,
	title = {Point detection through multi-instance deep heatmap regression for sutures in endoscopy},
	issn = {1861-6429},
	url = {https://doi.org/10.1007/s11548-021-02523-w},
	doi = {10.1007/s11548-021-02523-w},
	language = {en},
	urldate = {2021-11-16},
	journal = {International Journal of Computer Assisted Radiology and Surgery},
	author = {Sharan, Lalith and Romano, Gabriele and Brand, Julian and Kelm, Halvar and Karck, Matthias and De Simone,
	Raffaele and Engelhardt, Sandy},
	month = nov,
	year = {2021}
}
```

### Setup

A conda environment is recommended for setting up an environment for model training and prediction.
There are two ways this environment can be set up:

1. Cloning conda environment (recommended)
```
conda env create -f suture_detection_pytorch.yml
conda activate suture_detection_pytorch
```
If the installation from .yml file does not work, it may be a [cuda error](https://githubmemory.com/repo/markstrefford/running-detectron2-on-windows-wsl2-rtx30xx/issues/2?page=4).
The solution is to either install the failed packages via pip, or use the pip requirements file here.

2. Installing requirements
```
conda intall --file conda_requirements.txt
conda install -c pytorch torchvision=0.7.0
pip install --r requirements.txt
```

### Prediction of suture detection for a single image

You can predict the suture points for a single image with:
```shell
python test.py --dataroot ~/data/mkr_dataset/ --exp_dir ~/experiments/unet_baseline_fold_1/ --save_pred_points
```
* The command ```save_pred_points``` saves the predicted landmark co-ordinates in the resepective op folders in the ```../predictions``` directory.
* The command ```save_pred_mask``` saves the predicted mask that is the output of the model in the resepective op folders in the ```../predictions``` directory. The final points are extracted from this mask.

### Dataset preparation

You can download the challenge dataset from the synapse platform by signing up for the [AdaptOR 2021 Challenge](https://adaptor2021.github.io/) from the Synapse platform.
* The Challenge data is present in this format: dataroot --> op_date --> video_folders --> images, point_labels
* Generate the masks with a blur function and spread by running the following script:
```shell
python generate_suture_masks.py --dataroot /path/to/data --blur_func gaussian --spread 2
```

* Generate the split files for the generated masks, for cross-validation by running the following script:
You can predict depth for a single image with:
```shell
python generate_splits.py --splits_name mkr_dataset --num_folds 4
```

### Training a model

Once you have prepared the dataset, you can train the model with:

```shell
python train.py --dataroot /path/to/data
```

* This repo is inspired by the following repos:
* [CycleGAN PyTorch](https://github.com/aitorzip/PyTorch-CycleGAN)
* [Monodepth2](https://github.com/nianticlabs/monodepth2)
* [DetCycleGAN](https://github.com/Cardio-AI/detcyclegan_pytorch/blob/main/README.md)
