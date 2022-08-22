This is the codebase for our paper “Learning Multiple Probabilistic Degradation Generators for Unsupervised Real World Image Super Resolution", ECCVW 2022. Tested environment : Pytorch 1.7.1, cuda 10.1.



> **Learning Multiple Probabilistic Degradation Generators for Unsupervised Real World Image Super Resolution**<br>
> Sangyun Lee<sup>1</sup>, Sewoong Ahn<sup>2</sup>, Kwangjin Yoon<sup>2</sup>

> <sup>1</sup>Soongsil University, <sup>2</sup>SI Analytics<br>

> Paper: https://arxiv.org/abs/2201.10747<br>

> **Abstract:** *Unsupervised real world super resolution (USR) aims to restore high-resolution (HR) images given low-resolution (LR) inputs, and its difficulty stems from the absence of paired dataset. One of the most common approaches is synthesizing noisy LR images using GANs (i.e., degradation generators) and utilizing a synthetic dataset to train the model in a supervised manner. Although the goal of training the degradation generator is to approximate the distribution of LR images given a HR image, previous works have heavily relied on the unrealistic assumption that the conditional distribution is a delta function and learned the deterministic mapping from the HR image to a LR image. In this paper, we show that we can improve the performance of USR models by relaxing the assumption and propose to train the probabilistic degradation generator. Our probabilistic degradation generator can be viewed as a deep hierarchical latent variable model and is more suitable for modeling the complex conditional distribution. We also reveal the notable connection with the noise injection of StyleGAN. Furthermore, we train multiple degradation generators to improve the mode coverage and apply collaborative learning for ease of training. We outperform several baselines on benchmark datasets in terms of PSNR and SSIM and demonstrate the robustness of our method on unseen data distribution.*



## Stage 1 : Training probabilistic degradation generators

First, you need to install [pytorch_wavelet](https://github.com/fbcotter/pytorch_wavelets).

```bash
$ cd ./degradation/codes
$ git clone <https://github.com/fbcotter/pytorch_wavelets>
$ cd pytorch_wavelets
$ pip install .
```

Next, specify the directories to datasets in `MSSR/degradation/paths.yml`. Now, we are ready to train.

### DeResnet

`cd ./degradation/codes` and enter

```bash
# AIM2019
python3 train.py --gpu k --dataset aim2019 --artifact tdsr --batch_size 8 --flips --rotations --generator DeResnet --filter wavelet --save_path ./test --noise_std 0.1 --num_epoch 500

# NTIRE2020
python3 train.py --gpu k --dataset mydsn --artifact a --batch_size 8 --flips --rotations --generator DeResnet --filter wavelet --save_path ./test --noise_std 0.1 --num_epoch 500
```

### HAN

```bash
# AIM2019
python3 train.py --gpu k --dataset aim2019 --artifact tdsr --batch_size 4 --flips --rotations --generator han --filter wavelet --save_path YOUR_LOG_PATH --noise_std 0.1 --num_epoch 500 --learning_rate 0.00001

# NTIRE2020
python3 train.py --gpu k --dataset mydsn --artifact a --batch_size 4 --flips --rotations --generator han --filter wavelet --save_path YOUR_LOG_PATH --noise_std 0.1 --num_epoch 500 --learning_rate 0.00001
```

## Stage 2 : Training super resolution network

You should download the pre-trained ESRGAN checkpoint from [here](https://drive.google.com/file/d/1swaV5iBMFfg-DL6ZyiARztbhutDCWXMM/view?usp=sharing). After downloaded, go to `MSSR/sr/config_cont.json` and configure the overall training as well as the first degradation generator. Change `MSSR/sr/config_cont2.json` for the second degradation generator (the other arguments that you already specified in `MSSR/sr/config_cont.json`have no effect on training). Finally, enter

```
python3 contsr_train.py
```

# Acknowledgement

We build our code on [DASR](https://github.com/ShuhangGu/DASR) and [BasicSR](https://github.com/XPixelGroup/BasicSR) repositories.
