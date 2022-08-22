import torch
from utils import *
import model
import torchvision.transforms.functional as TF
import argparse
parser = argparse.ArgumentParser(description='Apply the trained model to create a dataset')
parser.add_argument('--gpu', type=str, help='gpu num')
parser.add_argument('--n_GPUs', default=1, type=int, help='ngpu')
parser.add_argument('--n_resgroups', default=10, type=int, help='n_resblock in HAN')
parser.add_argument('--neptune', default=False, type=bool, help='Whether to use neptune or not')
parser.add_argument('--use_noise', default=False, type=bool, help='use noise or not')
parser.add_argument('--noise_std', default=0.1, type=float, help='Noise std')


parser.add_argument('--checkpoint', default="/mnt/workspace/DASR/codes/DSN/save_data/checkpoints/deresnet/iteration_205531.tar", type=str, help='checkpoint model to use')
parser.add_argument('--generator', default='DeResnet', type=str, help='Generator model to use')
parser.add_argument('--num_res_blocks', default=8, type=int, help='number of ResNet blocks')
parser.add_argument('--discriminator', default='FSD', type=str, help='Discriminator model to use')
parser.add_argument('--kernel_size', default=5, type=int, help='kernel size used in transformation for discriminators')
parser.add_argument('--wgan', dest='wgan', action='store_true',
                    help='if activated then WGAN-GP is used instead of DCGAN')
parser.add_argument('--no_highpass', dest='highpass', action='store_false',
                    help='if activated then the highpass filter before the discriminator is omitted')
parser.add_argument('--filter', default='wavelet', type=str, help='set filter')
parser.add_argument('--cat_or_sum', default='cat', type=str, help='set wavelet bands type')
parser.add_argument('--norm_layer', default='Instance', type=str, help='set type of discriminator norm layer')
parser.add_argument('--artifacts', default='tdsr', type=str, help='selecting different artifacts type')
parser.add_argument('--name', default='0603_DSN_LRs', type=str, help='additional string added to folder path')
parser.add_argument('--dataset', default='aim2019', type=str, help='selecting different datasets')
parser.add_argument('--including_source_ddm', dest='including_source_ddm', action='store_true', help='generate ddm from '
                                                                                                     'source images')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[4, 1, 2], help='super resolution upscale factor')

opt = parser.parse_args()


# Build G
G = model.De_resnet(opt, n_res_blocks=opt.num_res_blocks, scale=opt.upscale_factor)

checkpoint = torch.load(opt.checkpoint, map_location='cpu')
G.load_state_dict(checkpoint['model_g_state_dict'])

y = pil_loader("/mnt/data/track1/DIV2K_valid_HR/0818.png")
y = TF.to_tensor(y)
y = torch.unsqueeze(y,dim=0)
#load y, unsqueeze y
fake = G(y)[0]
tensor_imsave(fake, "./noise_ablation3/", 'default.png')
# for std in [0]:
#     G.set_std(std, 5, 8)
#     fake = G(y)[0]
#     tensor_imsave(fake, "./noise_ablation2/", str(std) + '_last.png')

