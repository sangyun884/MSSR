import sys

sys.path.append("../")

import cv2
import numpy as np
import argparse
from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.registry import METRIC_REGISTRY
import utils
import torch
import os
import torchvision.transforms.functional as TF



def calculate_psnr(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='/data/private/ntire2020/track1-valid-clean')
parser.add_argument('-d1','--dir1', type=str, default='/data/private/MSSR/sr/nonprophan-inference/val')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--yonly', type=bool, default=False)

opt = parser.parse_args()
if opt.yonly:
    opt.out = os.path.join(opt.dir1, "result_y.txt")
else:
    opt.out = os.path.join(opt.dir1, "result_rgb.txt")
f = open(opt.out,'w')
files = os.listdir(opt.dir0)
files.sort()
prefix= opt.prefix
m_list = []
print(files)
psnr_list = []
ssim_list = []

for file in files:
    if(os.path.exists(os.path.join(opt.dir1, prefix+file))):
        print(os.path.join(opt.dir1,prefix+file))

        img0 = utils.pil_loader(os.path.join(opt.dir0,file))
        img1 = utils.pil_loader(os.path.join(opt.dir1,prefix+file))
        img0 = np.asarray(img0)
        img1 = np.asarray(img1)
    
        psnr = calculate_psnr(img0,img1,crop_border=4, test_y_channel=opt.yonly)
      
        ssim = calculate_ssim(img0,img1, crop_border=4, test_y_channel=opt.yonly)

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        f.writelines('%s: psnr : %.6f, ssim : %.6f \n'%(file,psnr, ssim))
avg_ssim = sum(ssim_list)/len(ssim_list)
avg_psnr = sum(psnr_list)/len(psnr_list)
f.writelines(f"AVG_ssim : {avg_ssim}")
f.writelines(f"AVG_psnr : {avg_psnr}")     
f.close()
print(avg_ssim, avg_psnr)

# from PIL import Image
# img0 = utils.pil_loader(os.path.join("/mnt/data/track1/DIV2K_valid_HR","0853.png"))
# h = img0.size[0]
# w = img0.size[1]
# img1 = img0.resize((h//4,w//4), Image.BICUBIC).resize((h,w), Image.BICUBIC)
# img0 = np.asarray(img0)
# img1 = np.asarray(img1)

# psnr = calculate_psnr(img0,img1, crop_border=4, test_y_channel=False)
# print(psnr)


# from PIL import Image
# files = os.listdir('/mnt/data/track1/DIV2K_valid_HR')
# files.sort()
# psnr_list = []
# for file in files:
#     img0 = utils.pil_loader(os.path.join("/mnt/data/track1/DIV2K_valid_HR",file))
#     h = img0.size[0]
#     w = img0.size[1]
#     img1 = img0.resize((h//4,w//4), Image.BICUBIC).resize((h,w), Image.BICUBIC)
#     img0 = np.asarray(img0)
#     img1 = np.asarray(img1)

#     psnr = calculate_psnr(img0,img1, crop_border=4, test_y_channel=True)
#     psnr_list.append(psnr)
#     print(psnr)
# print(sum(psnr_list)/len(psnr_list))
