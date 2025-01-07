# Self-Supervised Learning for Detecting AI-Generated Faces as Anomalies
This repository contains the official PyTorch implementation of the paper **"Self-Supervised Learning for Detecting AI-Generated Faces as Anomalies"** by Mian Zou, Baosheng Yu, Yibing Zhan, and Kede Ma, in *ICASSP*, 2025.

☀️ If you find this work useful for your research, please kindly star our repo and cite our paper! ☀️

### TODO
We are working hard on the following items.

- [x] Release [arXiv paper](https://arxiv.org/abs/2501.02207)
- [ ] Release training codes
- [ ] Release inference codes
- [ ] Release checkpoints 
- [ ] Release datasets

## Introduction
In this paper, we describe an anomaly detection method for AI-generated faces by leveraging self-supervised learning of camera-intrinsic and face-specific features purely from photographic face images. The success of our method lies in designing a pretext task that trains a feature extractor to rank four ordinal exchangeable image file format (EXIF) tags and classify artificially manipulated face images. Subsequently, we model the learned feature distribution of photographic face images using a Gaussian mixture model. Faces with low likelihoods are flagged as AI-generated.

![IMG_00001](https://github.com/MZMMSEC/AIGFD_EXIF/blob/50ed5f5deb1f9d20b28869fff87917fd50f4adb1/imgs/framework.jpg)

## Citation
If you find this repository useful in your research, please consider citing the following paper:
```
@inproceedings{zou2025self,
    title={Self-Supervised Learning for Detecting AI-Generated Faces as Anomalies},
    author={Mian Zou and Baosheng Yu and Yibing Zhan and Kede Ma},
    booktitle={ICASSP},
    year={2025}
}
```
