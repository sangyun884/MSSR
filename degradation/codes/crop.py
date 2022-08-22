import utils

# std_list = [0,0.02,0.04,0.06,0.08,0.1]
# paths = [f"/mnt/workspace/DASR/codes/DSN/noise_ablation/{x}.png" for x in std_list]
# for std,path in zip(std_list,paths):
#     img = utils.pil_loader(path)
#     img_cropped = img.crop((120,10,200,90))
#     img_cropped.save(f"/mnt/workspace/DASR/codes/DSN/noise_ablation/{std}_cropped.png")
# for std in [0]:
#     img = utils.pil_loader(f"/mnt/data/Real-SR_NTIRE2020_iter20000/0804.png")
#     k = 150
#     img_cropped = img.crop((180,180,180+k,180+k))
#     img_cropped.save(f"/mnt/workspace/DASR/codes/DSN/robustness/imp_{std}.png")

import os
# num = 896
# simusr_path = f'/mnt/workspace/imagenetEDSR/MSSR-85/inference/0{num}.png'
# mssr_path = f'/mnt/workspace/BasicSR/basicsr/archs/MSSR-110/inference/val/ensemble_0{num}.png'
# dasr_path = f'/mnt/workspace/DASR/SRN_experiments/experiments/06013_DASR_SRN_auto_reproduce_aim2019/val_images/best_result/0{num}.png'
# rrdb_path = f'/mnt/workspace/BasicSR/basicsr/archs/RRDB_aim_inference/val/g1_0{num}.png'
# zssr_path = f'/mnt/workspace/pytorch-zssr/aim_inference_10000iter/0{num}.png'
# fssr_path = f'/mnt/data/FSSR_AIM2019_iter60000/0{num}.png'
# gt_path = f'/mnt/data/track1/DIV2K_valid_HR/0{num}.png'

# paths = [[simusr_path,'simusr'], [mssr_path,'mssr'], [dasr_path,'dasr'], [rrdb_path,'rrdb'], [zssr_path,'zssr'], [gt_path,'gt'], [fssr_path,'fssr']]

# crop_x = 1218
# crop_y = 561
# size = 150

# for path, model in paths:
#     img = utils.pil_loader(path)
#     fname = path.split('/')[-1]
#     print(f"path : {path}, fname : {fname}")
#     img_cropped = img.crop((crop_x,crop_y,crop_x+size,crop_y+size))
#     img_cropped.save(os.path.join("/mnt/workspace/DASR/codes/DSN/comp_aim", model + "_"+fname))


noise_std = [2.5,5,7.5,10]
# x = 188
# y = 167
# size_ = 150
# num = 804

# save_path = "/mnt/workspace/DASR/codes/DSN/robustness2"
# for std in noise_std:
#     path_mssr = f"/mnt/workspace/BasicSR/basicsr/archs/MSSR-65/inference/{std}/val"
#     path_fssr = f"/mnt/data/FSSR_NTIRE2020_std{std}"
#     path_imp = f"/mnt/data/Real-SR_NTIRE2020_std{std}"
#     path_simusr = f"/mnt/workspace/imagenetEDSR/MSSR-83/inference_{std}"
#     paths = [[path_simusr,'simusr'], [path_mssr,'mssr'], [path_fssr,'fssr'], [path_imp, 'impressionism']]
    
#     img_mssr = utils.pil_loader(os.path.join(path_mssr, f"ensemble_0{num}.png"))
#     img_fssr = utils.pil_loader(os.path.join(path_fssr, f"0{num}.png"))
#     img_simusr = utils.pil_loader(os.path.join(path_simusr, f"0{num}.png"))
#     img_imp = utils.pil_loader(os.path.join(path_imp, f"0{num}.png"))

#     img_mssr = img_mssr.crop((x,y,x+size_,y+size_))
#     img_fssr = img_fssr.crop((x,y,x+size_,y+size_))
#     img_simusr = img_simusr.crop((x,y,x+size_,y+size_))
#     img_imp = img_imp.crop((x,y,x+size_,y+size_))

#     img_mssr.save(os.path.join(save_path,f"mssr_{std}.png"))
#     img_fssr.save(os.path.join(save_path,f"fssr_{std}.png"))
#     img_simusr.save(os.path.join(save_path,f"simusr_{std}.png"))
#     img_imp.save(os.path.join(save_path,f"imp_{std}.png"))
    
    
    

x = 0
y = 0
size_ = 100
num = '0006'
import numpy as np
from PIL import Image
save_path = "/data/private/MSSR/degradation/codes/diverse_LR"
img_path = "/data/private/MSSR/DSN_results/prophan"
img_arr = []
for i in [1,2,3]:
    path = img_path + str(i) + "/imgs_from_target/"
    img = utils.pil_loader(os.path.join(path, f"{num}.png"))
    img = img.crop((x,y,x+size_,y+size_))
    img.save(os.path.join(save_path,f"prophan_{num}_{i}.png"))
    img_arr.append(np.array(img))
diff1 = np.abs(img_arr[0]-img_arr[1]).sum(axis=-1)
diff2 = np.abs(img_arr[0]-img_arr[2]).sum(axis=-1)
print(diff1.max())
diff1_img = Image.fromarray(np.uint8(diff1))
diff2_img = Image.fromarray(np.uint8(diff2))

diff1_img.save(os.path.join(save_path,f"prophan_diff1.png"))
diff2_img.save(os.path.join(save_path,f"prophan_diff2.png"))

# for std in noise_std:
#     path_mssr = f"/mnt/workspace/BasicSR/basicsr/archs/MSSR-65/inference/{std}/val"
#     path_fssr = f"/mnt/data/FSSR_NTIRE2020_std{std}"
#     path_imp = f"/mnt/data/Real-SR_NTIRE2020_std{std}"
#     path_simusr = f"/mnt/workspace/imagenetEDSR/MSSR-83/inference_{std}"
#     paths = [[path_simusr,'simusr'], [path_mssr,'mssr'], [path_fssr,'fssr'], [path_imp, 'impressionism']]
    
#     img_mssr = utils.pil_loader(os.path.join(path_mssr, f"ensemble_0{num}.png"))
#     img_fssr = utils.pil_loader(os.path.join(path_fssr, f"0{num}.png"))
#     img_simusr = utils.pil_loader(os.path.join(path_simusr, f"0{num}.png"))
#     img_imp = utils.pil_loader(os.path.join(path_imp, f"0{num}.png"))

#     img_mssr = img_mssr.crop((x,y,x+size_,y+size_))
#     img_fssr = img_fssr.crop((x,y,x+size_,y+size_))
#     img_simusr = img_simusr.crop((x,y,x+size_,y+size_))
#     img_imp = img_imp.crop((x,y,x+size_,y+size_))

#     img_mssr.save(os.path.join(save_path,f"mssr_{std}.png"))
#     img_fssr.save(os.path.join(save_path,f"fssr_{std}.png"))
#     img_simusr.save(os.path.join(save_path,f"simusr_{std}.png"))
#     img_imp.save(os.path.join(save_path,f"imp_{std}.png"))
    
    


