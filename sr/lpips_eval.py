import argparse
import os
import lpips
import utils2
import torch
# import psnr_ssim
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./imgs/ex_dir0')
parser.add_argument('-d1','--dir1', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--gpu', type=int, help='turn on flag to use GPU. -1 means cpu')

opt = parser.parse_args()
device = torch.device('cpu' if opt.gpu==-1 else 'cuda:' + str(opt.gpu))
print("model init")
## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)

loss_fn.to(device)
opt.out = os.path.join(opt.dir1, 'result.txt')
# crawl directories
f = open(opt.out,'w')
files = os.listdir(opt.dir0)
files.sort()
prefix = ""
m_list = []
print(files)
psnr = utils2.PSNR(device=device, val_max = 1, val_min=-1)
psnr_list = []
for file in files:
    
    if(os.path.exists(os.path.join(opt.dir1, prefix+file))):
        # Load images
        print(os.path.join(opt.dir1,prefix+file))
        img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))).to(device) # RGB image from [-1,1]
        img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,prefix+file))).to(device)


        # Compute distance
        dist01 = loss_fn.forward(img0,img1)
        dist_psnr = psnr(img0,img1)
        m_list.append(dist01)
        psnr_list.append(dist_psnr)
        print('%s: %.3f, psnr:%.3f'%(file,dist01,dist_psnr))
        f.writelines('%s: %.6f, psnr : %.6f\n'%(file,dist01, dist_psnr))
avg = sum(m_list)/len(m_list)
avg_psnr = sum(psnr_list)/len(psnr_list)
f.writelines(f"AVG : {avg}")
f.writelines(f"AVG_psnr : {avg_psnr}")

f.close()
print(avg, avg_psnr)
