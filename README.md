Pytorch implementation of "Unsupervised Real World Image Super Resolution with Multi-Source Collaborative Learning and Noise Injection", BMVC2021 **(in review)**.

## Stage 1 : Degradation
### DeResnet
cd ./degradation/codes
```python
python3 train.py --gpu k --dataset aim2019 --artifact tdsr --batch_size 8 --flips --rotations --generator DeResnet --filter wavelet --save_path YOUR_LOG_PATH --noise_std 0.1 --num_epoch 500

```
### HAN

Refer to MSSR/degradation/paths.yml for dataset configuration.

```python
python3 train.py --gpu k --dataset aim2019 --artifact tdsr --batch_size 4 --flips --rotations --generator han --filter wavelet --save_path YOUR_LOG_PATH --noise_std 0.1 --num_epoch 500 --learning_rate 0.00001

```

## Stage 2 : Super Resolution

cd ./sr
```python
python3 contsr_train.py

```

Refer to MSSR/sr/config_cont.json for configuration of the overall training and first degradation generator. Change MSSR/sr/config_cont2.json for the second degradation generator.

Tested environment : Pytorch 1.7.1, cuda 10.1.

Pre-trained ESRGAN checkpoint: https://drive.google.com/file/d/1swaV5iBMFfg-DL6ZyiARztbhutDCWXMM/view?usp=sharing
