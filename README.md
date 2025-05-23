# Self-Supervised Learning for Detecting AI-Generated Faces as Anomalies
This repository contains the official PyTorch implementation of the paper **"Self-Supervised Learning for Detecting AI-Generated Faces as Anomalies"** by Mian Zou, Baosheng Yu, Yibing Zhan, and Kede Ma.

☀️ If you find this work useful for your research, please kindly star our repo and cite our paper! ☀️

### TODO
We are working hard on the following items.

- [x] Release [arXiv paper](https://arxiv.org/abs/2501.02207)
- [ ] Release training codes
- [x] Release inference codes
- [x] Release checkpoints 
- [x] Release datasets

## Introduction
In this paper, we describe an anomaly detection method for AI-generated faces by leveraging self-supervised learning of camera-intrinsic and face-specific features purely from photographic face images. The success of our method lies in designing a pretext task that trains a feature extractor to rank four ordinal exchangeable image file format (EXIF) tags and classify artificially manipulated face images. Subsequently, we model the learned feature distribution of photographic face images using a Gaussian mixture model. Faces with low likelihoods are flagged as AI-generated.

![IMG_00001](https://github.com/MZMMSEC/AIGFD_EXIF/blob/50ed5f5deb1f9d20b28869fff87917fd50f4adb1/imgs/framework.jpg)

## 🚀 Quick Start

### 1. Installation of base reqiurements
 - python == 3.8
 - PyTorch == 1.13
 - Miniconda
 - CUDA == 11.7

### 2. Download the pretrained model and our model

|      Model       |                                                               Download                                                                |
|:----------------:|:-------------------------------------------------------------------------------------------------------------------------------------:|
| EXIF-pretrained | [Google Drive](https://drive.google.com/file/d/17MW-fZRRQQ8dSRv52X_9DmcmdQD7TmHZ/view?usp=drive_link) |
| our model    | [Google Drive](https://drive.google.com/file/d/1rpKta773mA-hgKOoycZODUAbNpDUyZ8f/view?usp=sharing)                  |

### 3. Inference on the test sets

```
CUDA_VISIBEL_DEVICES=XXX python yfcc_face_OCT_variants.py --resume [path to our model checkpoints]
```

## 📁 Datasets

We use FDF (CC BY-NC-SA 2.0 version) for self-supervised training, and you can download it from the original project page.

For the test sets we used in the main experiments, we collected them from [DiffusionFace](https://github.com/Rapisurazurite/DiffFace) and [DiFF](https://github.com/xaCheng1996/DiFF), and put them together for testing. If you find them useful, please cite these two papers.

| Dataset |                                                 Link                                                 |
|:-------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------:|
|FDF| [GitHub](https://github.com/hukkelas/FDF) |
|Test Sets| [Baidu Disk]() |


During testing, please put the dataset in the ``data`` folder, and the data structure is as follows:
```
data
├── celeba-face
├── DDIM
├── FreeDoM_T
├── HPS
├── LDM
├── Midjourney
├── SDXL
├── stable_diffusion_v_2_1_text2img_p2g3
├── stylegan2
├── VQ-GAN_celebahq

```


## Citation
If you find this repository useful in your research, please consider citing the following paper:
```
@article{zou2025self,
  title={Self-Supervised Learning for Detecting AI-Generated Faces as Anomalies},
  author={Zou, Mian and Yu, Baosheng and Zhan, Yibing and Ma, Kede},
  journal={arXiv preprint arXiv:2501.02207},
  year={2025}
}
```
