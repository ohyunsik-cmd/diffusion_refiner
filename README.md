<div align="center">
<h1>Diffusion Refiner using VGGT</h1>

## Overview
For novel view synthesis, we render the VGGT point cloud into the target view. A diffusion model is then employed to refine the resulting sparse images into high-quality renderings.


## Quick Start

First, clone this repository to your local machine, and install the dependencies (torch, torchvision, numpy, Pillow, and huggingface_hub). 

```bash
git clone git@https://github.com/ohyunsik-cmd/diffusion_refiner
cd diffusion_refiner
pip install -r requirements_full.txt
```

## Dataset folder

We use the RE10K dataset for training. You must specify the dataset path as a command-line argument when starting the training process. The dataset directory should be structured as follows:

``` 
re10k_DIR/
├── train/
├── ├── 000000.torch
├── ├── ...
├── ├── index.json
└── test/
    ├── 000000.torch
    ├── ...
    └── index.json
```

## Train diffusion with freeze VGGT and Difix3D+ pretrained model 

The Difix3D+ model fine-tunes the SD-Turbo model into a refiner. The UNet was fully fine-tuned, while the encoder remained frozen. The decoder was trained using LoRA adapters, and skip connections were established between the encoder and the decoder. All weights are resumed, and this code allows for training the LoRA adapters exclusively.

An example command to train the model is:
```
CUDA_VISIBLE_DEVICES=2 accelerate launch --mixed_precision=bf16 train_difix.py \
  --output_dir ./outputs/difix_train \
  --tracker_project_name difix \
  --tracker_run_name difix_re10k_bf16 \
  --re10k_root /mnt/hdd1/yunsik/re10k \
  --image_size 336 \
  --train_batch_size 4 \
  --max_train_steps 10000 \
  --learning_rate 2e-5 \
  --timestep 199
```


## Acknowledgements

Thanks to these great repositories: [PoseDiffusion](https://github.com/facebookresearch/PoseDiffusion), [VGGSfM](https://github.com/facebookresearch/vggsfm), [CoTracker](https://github.com/facebookresearch/co-tracker), [DINOv2](https://github.com/facebookresearch/dinov2), [Dust3r](https://github.com/naver/dust3r), [Moge](https://github.com/microsoft/moge), [PyTorch3D](https://github.com/facebookresearch/pytorch3d), [Sky Segmentation](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), [Metric3D](https://github.com/YvanYin/Metric3D), [VGGT](https://github.com/facebookresearch/vggt), [Difix3D](https://github.com/nv-tlabs/Difix3D) and many other inspiring works in the community.

## Checklist

- [x] made the training code with freeze VGGT model
- [ ] made the training code with freeze spfsplatv2 model


## License
no license still
