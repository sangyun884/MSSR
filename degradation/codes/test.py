import argparse
import os
import torch.optim as optim
import torch.utils.data
import torchvision.utils as tvutils
import data_loader as loader
import yaml
import loss
import model
from receptive_cal import *
import utils
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from PIL import Image
from pytorch_wavelets import DWTForward
import hanmodel

def saveimgs(img_list, img_name, savepath):
    img = img_list[0][0].cpu().numpy().transpose((1,2,0))
    img = Image.fromarray(np.uint8(img*255))
    img.save(savepath+img_name.split('.')[0]+'.png')
    print(savepath+img_name.split('.')[0]+'.png')

    # img = img_list[1][0].cpu().numpy().transpose((1,2,0))
    # img = Image.fromarray(np.uint8(img*255))
    # img.save(savepath+img_name+img_name[1]+'_blur.png')
    # print(savepath + img_name + img_name[1]+'_blur.png')
    #
    # img = img_list[2][0].cpu().numpy().transpose((1,2,0))
    # img = Image.fromarray(np.uint8(img*255))
    # img.save(savepath+img_name+img_name[2]+'_hf.png')
    # print(savepath+img_name+img_name[2]+'_hf.png')


# parser = argparse.ArgumentParser(description='Train Downscaling Models')
# parser.add_argument('--upscale_factor', default=4, type=int, choices=[4], help='super resolution upscale factor')
# parser.add_argument('--crop_size', default=512, type=int, help='training images crop size')
# parser.add_argument('--crop_size_val', default=256, type=int, help='validation images crop size')
# parser.add_argument('--batch_size', default=16, type=int, help='batch size used')
# parser.add_argument('--num_workers', default=4, type=int, help='number of workers used')
# parser.add_argument('--num_epochs', default=300, type=int, help='total train epoch number')
# parser.add_argument('--num_decay_epochs', default=150, type=int, help='number of epochs during which lr is decayed')
# parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate')
# parser.add_argument('--adam_beta_1', default=0.5, type=float, help='beta_1 for adam optimizer of gen and disc')
# parser.add_argument('--val_interval', default=1, type=int, help='validation interval')
# parser.add_argument('--val_img_interval', default=30, type=int, help='interval for saving validation images')
# parser.add_argument('--save_model_interval', default=30, type=int, help='interval for saving the model')
# parser.add_argument('--artifacts', default='gaussian', type=str, help='selecting different artifacts type')
# parser.add_argument('--dataset', default='df2k', type=str, help='selecting different datasets')
# parser.add_argument('--flips', dest='flips', action='store_true', help='if activated train images are randomly flipped')
# parser.add_argument('--rotations', dest='rotations', action='store_true',
#                     help='if activated train images are rotated by a random angle from {0, 90, 180, 270}')
# parser.add_argument('--num_res_blocks', default=8, type=int, help='number of ResNet blocks')
# parser.add_argument('--ragan', dest='ragan', action='store_true',
#                     help='if activated then RaGAN is used instead of normal GAN')
# parser.add_argument('--wgan', dest='wgan', action='store_true',
#                     help='if activated then WGAN-GP is used instead of DCGAN')
# parser.add_argument('--no_highpass', dest='highpass', action='store_false',
#                     help='if activated then the highpass filter before the discriminator is omitted')
# parser.add_argument('--kernel_size', default=5, type=int, help='kernel size used in transformation for discriminators')
# parser.add_argument('--gaussian', dest='gaussian', action='store_true',
#                     help='if activated gaussian filter is used instead of average')
# parser.add_argument('--no_per_loss', dest='use_per_loss', action='store_false',
#                     help='if activated no perceptual loss is used')
# parser.add_argument('--lpips_rot_flip', dest='lpips_rot_flip', action='store_true',
#                     help='if activated images are randomly flipped and rotated before being fed to lpips')
# parser.add_argument('--disc_freq', default=1, type=int, help='number of steps until a discriminator updated is made')
# parser.add_argument('--gen_freq', default=1, type=int, help='number of steps until a generator updated is made')
# parser.add_argument('--w_col', default=1, type=float, help='weight of color loss')
# parser.add_argument('--w_tex', default=0.005, type=float, help='weight of texture loss')
# parser.add_argument('--w_per', default=0.01, type=float, help='weight of perceptual loss')
# parser.add_argument('--checkpoint', default=None, type=str, help='checkpoint model to start from')
# parser.add_argument('--save_path', default=None, type=str, help='additional folder for saving the data')
# parser.add_argument('--no_saving', dest='saving', action='store_false',
#                     help='if activated the model and results are not saved')
# parser.add_argument('--val_image_path', default=None, type=str, help='checkpoint model to start from')

parser = argparse.ArgumentParser(description='Train Downscaling Models')
parser.add_argument('--gpu', type=str, help='gpu num')
parser.add_argument('--n_GPUs', default=1, type=int, help='ngpu')
parser.add_argument('--n_resgroups', default=10, type=int, help='n_resblock in HAN')

parser.add_argument('--upscale_factor', default=4, type=int, choices=[1, 2, 4], help='super resolution upscale factor')
parser.add_argument('--crop_size', default=256, type=int, help='training images crop size')
parser.add_argument('--crop_size_val', default=256, type=int, help='validation images crop size')
parser.add_argument('--batch_size', default=4, type=int, help='batch size used')
parser.add_argument('--num_workers', default=6, type=int, help='number of workers used')
parser.add_argument('--num_epochs', default=400, type=int, help='total train epoch number')
parser.add_argument('--num_decay_epochs', default=150, type=int, help='number of epochs during which lr is decayed')
parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate')
parser.add_argument('--adam_beta_1', default=0.5, type=float, help='beta_1 for adam optimizer of gen and disc')
parser.add_argument('--val_interval', default=1, type=int, help='validation interval')
parser.add_argument('--val_img_interval', default=1, type=int, help='interval for saving validation images')
parser.add_argument('--save_model_interval', default=5, type=int, help='interval for saving the model')
parser.add_argument('--artifacts', default='tdsr', type=str, help='selecting different artifacts type')
parser.add_argument('--dataset', default='df2k', type=str, help='selecting different datasets')
parser.add_argument('--flips', dest='flips', action='store_true', help='if activated train images are randomly flipped')
parser.add_argument('--rotations', dest='rotations', action='store_true',
                    help='if activated train images are rotated by a random angle from {0, 90, 180, 270}')
parser.add_argument('--num_res_blocks', default=8, type=int, help='number of ResNet blocks')
parser.add_argument('--ragan', dest='ragan', action='store_true',
                    help='if activated then RaGAN is used instead of normal GAN')
parser.add_argument('--wgan', dest='wgan', action='store_true',
                    help='if activated then WGAN-GP is used instead of DCGAN')
parser.add_argument('--no_highpass', dest='highpass', action='store_false',
                    help='if activated then the highpass filter before the discriminator is omitted')
parser.add_argument('--kernel_size', default=5, type=int, help='kernel size used in transformation for discriminators')
parser.add_argument('--no_per_loss', dest='use_per_loss', action='store_false',
                    help='if activated no perceptual loss is used')
parser.add_argument('--lpips_rot_flip', dest='lpips_rot_flip', action='store_true',
                    help='if activated images are randomly flipped and rotated before being fed to lpips')
parser.add_argument('--per_type', default='LPIPS', type=str, help='selecting different Perceptual loss type')
parser.add_argument('--disc_freq', default=1, type=int, help='number of steps until a discriminator updated is made')
parser.add_argument('--gen_freq', default=1, type=int, help='number of steps until a generator updated is made')
parser.add_argument('--w_col', default=1, type=float, help='weight of color loss')
parser.add_argument('--w_tex', default=0.005, type=float, help='weight of texture loss')
parser.add_argument('--w_per', default=0.01, type=float, help='weight of perceptual loss')
parser.add_argument('--checkpoint', default=None, type=str, help='checkpoint model to start from')
parser.add_argument('--save_path', default=None, type=str, help='additional folder for saving the data')
parser.add_argument('--generator', default='han', type=str, help='set generator architecture. (DSGAN, DeResnet, han)')
parser.add_argument('--discriminator', default='FSD', type=str,
                    help='set discriminator architecture. (FSD, nld_s1, nld_s2)')
parser.add_argument('--filter', default='gau', type=str, help='set filter in (gau, avg_pool, wavelet)')
parser.add_argument('--cat_or_sum', default='cat', type=str, help='set wavelet bands type in (cat, sum)')
parser.add_argument('--norm_layer', default='Instance', type=str,
                    help='set type of discriminator norm layer in (Instance, Batch)')
parser.add_argument('--no_saving', dest='saving', action='store_false',
                    help='if activated the model and results are not saved')
parser.add_argument('--debug', dest='debug', action='store_true',
                    help='Activate to ensure whole pipeline works well')
parser.add_argument('--neptune', default=False, type=bool, help='Whether to use neptune or not')



opt = parser.parse_args()

# fix random seeds
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# # prepare neural networks
# model_g = model.Generator(n_res_blocks=opt.num_res_blocks)
# print('# generator parameters:', sum(param.numel() for param in model_g.parameters()))
# model_d = model.Discriminator(kernel_size=opt.kernel_size, filter_type=opt.filter, wgan=opt.wgan, highpass=opt.highpass)
# print('# discriminator parameters:', sum(param.numel() for param in model_d.parameters()))
# model_d_save = model.NLayerDiscriminator(3, n_layers=2)
# DWT2 = DWTForward(J=1, mode='zero', wave='haar').cuda()

# g_loss_module = loss.GeneratorLoss(**vars(opt))

# prepare neural networks
if opt.generator.lower() == 'dsgan':
    model_g = model.Generator(n_res_blocks=opt.num_res_blocks)
elif opt.generator.lower() == 'deresnet':
    model_g = model.De_resnet(n_res_blocks=opt.num_res_blocks, scale=opt.upscale_factor)
elif opt.generator.lower() == 'han':
    model_g = hanmodel.Model(opt, None)

else:
    raise NotImplementedError('Generator model [{:s}] not recognized'.format(opt.generator))
print('# Initializing {}'.format(opt.generator))
print('# generator parameters:', sum(param.numel() for param in model_g.parameters()), '\n')

model_d = model.Discriminator(kernel_size=opt.kernel_size, wgan=opt.wgan, highpass=opt.highpass,
                              D_arch=opt.discriminator, norm_layer=opt.norm_layer, filter_type=opt.filter,
                              cs=opt.cat_or_sum)
print('# discriminator parameters:', sum(param.numel() for param in model_d.parameters()))
g_loss_module = loss.GeneratorLoss(**vars(opt))

# filters are used for generating validation images
# filter_low_module = model.FilterLow(kernel_size=opt.kernel_size, gaussian=opt.gaussian, include_pad=False)
# filter_high_module = model.FilterHigh(kernel_size=opt.kernel_size, gaussian=opt.gaussian, include_pad=False)
# if torch.cuda.is_available():
#     model_g = model_g.cuda()
#     model_d = model_d.cuda()
#     model_d_save = model_d_save.cuda()
#     filter_low_module = filter_low_module.cuda()
#     filter_high_module = filter_high_module.cuda()


# filters are used for generating validation images
filter_low_module = model.FilterLow(kernel_size=opt.kernel_size, gaussian=opt.filter == 'gau', include_pad=False)
filter_high_module = model.FilterHigh(kernel_size=opt.kernel_size, gaussian=opt.filter == 'gau', include_pad=False)
print('# FS type: {}, kernel size={}'.format(opt.filter, opt.kernel_size))
if torch.cuda.is_available():
    model_g = model_g.cuda()
    model_d = model_d.cuda()
    filter_low_module = filter_low_module.cuda()
    filter_high_module = filter_high_module.cuda()


# # define optimizers
# optimizer_g = optim.Adam(model_g.parameters(), lr=opt.learning_rate, betas=[opt.adam_beta_1, 0.999])
# optimizer_d = optim.Adam(model_d.parameters(), lr=opt.learning_rate, betas=[opt.adam_beta_1, 0.999])
# start_decay = opt.num_epochs - opt.num_decay_epochs
# scheduler_rule = lambda e: 1.0 if e < start_decay else 1.0 - max(0.0, float(e - start_decay) / opt.num_decay_epochs)
# scheduler_g = optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=scheduler_rule)
# scheduler_d = optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=scheduler_rule)

# load/initialize parameters
if opt.checkpoint is not None:
    checkpoint = torch.load(opt.checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    iteration = checkpoint['iteration'] + 1
    model_g.load_state_dict(checkpoint['model_g_state_dict'])
    # model_d.load_state_dict(checkpoint['models_d_state_dict'])
    # optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    # optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    # scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
    # scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
    print('Continuing training at epoch %d' % start_epoch)
else:
    start_epoch = 1
    iteration = 1

model_g.eval()
val_images = []

opt.val_image_path = '/mnt/data/track1/DIV2K_valid_HR/'
with torch.no_grad():
    # initialize variables to estimate averages
    mse_sum = psnr_sum = rgb_loss_sum = mean_loss_sum = 0
    per_loss_sum = col_loss_sum = tex_loss_sum = 0

    # validate on each image in the val dataset
    for input_imgname in os.listdir(opt.val_image_path):
        img_path = opt.val_image_path + input_imgname
        input_img = torch.from_numpy(np.ascontiguousarray(np.transpose(np.array(Image.open(img_path)) / 255, (2, 0, 1)))).float()
        input_img = input_img.reshape([1]+list(input_img.shape))
        if torch.cuda.is_available():
            input_img = input_img.cuda()
        fake_img = torch.clamp(model_g(input_img), min=0, max=1)


        # generate images
        # blur = filter_low_module(fake_img)
        hf = filter_high_module(fake_img)
        saveimgs([fake_img], input_imgname, opt.save_path)
        saveimgs([hf], "hf_wav_"+input_imgname, opt.save_path)
        
