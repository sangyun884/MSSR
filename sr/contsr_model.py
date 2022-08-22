import rrdbnet_arch
import torch
from dataset import *
from utils import *
from torchvision import transforms
from transforms import *
from torch.utils.data import DataLoader
import time
from torch import nn
import lpips
import model
import hanmodel
import rcanmodel
import torchvision.transforms.functional as TF
import edsr
class ContSR():
    def __init__(self, args, args2):
        self.args = args
        self.args2 = args2
        
        
    
    def build(self):
        print("BUILD")
        if self.args.backbone=='rrdb':
            self.g1 = rrdbnet_arch.RRDBNet(3,3).cuda(self.args.gpu)
            self.g2 = rrdbnet_arch.RRDBNet(3,3).cuda(self.args.gpu)
        elif self.args.backbone=='rcan':
            self.g1 = rcanmodel.Model(self.args)
            self.g2 = rcanmodel.Model(self.args)
        elif self.args.backbone=='edsr':
            self.g1 = edsr.EDSR(gpu=self.args.gpu, scale_factor=self.args.scale_factor).cuda(self.args.gpu)
            self.g2 = edsr.EDSR(gpu=self.args.gpu, scale_factor=self.args.scale_factor).cuda(self.args.gpu)
            
        
        train_target_transform = transforms.Compose([
            transforms.RandomCrop((self.args.lrsize,self.args.lrsize)),
            transforms.ToTensor()
        ])

        if self.args.generator == 'DSGAN':
            model_g2 = model.Generator(n_res_blocks=self.args.num_res_blocks)
        elif self.args.generator == 'DeResnet':
            model_g2 = model.De_resnet(self.args, n_res_blocks=self.args.num_res_blocks, scale=self.args.scale_factor)
        elif self.args.generator.lower() == 'han' or self.args.generator.lower() == 'han2':
            model_g2 = hanmodel.Model(self.args, None)
        else:
            raise NotImplementedError('Generator model [{:s}] not recognized'.format(opt.generator))
        self.model_g2 = model_g2.cuda(self.args.gpu)

        if self.args2.generator == 'DSGAN':
            model_g1 = model.Generator(n_res_blocks=self.args2.num_res_blocks)
        elif self.args2.generator == 'DeResnet':
            model_g1 = model.De_resnet(self.args2, n_res_blocks=self.args2.num_res_blocks, scale=self.args2.scale_factor)
        elif self.args2.generator.lower() == 'han' or self.args2.generator.lower() == 'han2':
            model_g1 = hanmodel.Model(self.args2, None)
        else:
            raise NotImplementedError('Generator model [{:s}] not recognized'.format(self.args2.generator))
        self.model_g1 = model_g1.cuda(self.args.gpu)



        self.dataset_tr = TwoSourceDatasetFolder(self.args.train_s_path1, self.args.train_s_path2, self.args.train_label_path, 0, lrsize = self.args.lrsize, scale_factor = self.args.scale_factor, random=self.args.aug)
        self.dataset_tr_target = DatasetFolder(self.args.train_target_path, 0, transform=train_target_transform)
        self.dataset_test = TestDatasetFolder(self.args.test_x_path, self.args.test_y_path, 0)

        self.loader_tr = DataLoader(self.dataset_tr, batch_size=self.args.batch_size, shuffle=True, num_workers = self.args.num_workers)
        self.loader_tr_target = DataLoader(self.dataset_tr_target, batch_size=self.args.batch_size, shuffle=True, num_workers = self.args.num_workers)
        self.loader_test = DataLoader(self.dataset_test, batch_size=1, shuffle=False, num_workers = self.args.num_workers)

        self.g1_optim = torch.optim.Adam(self.g1.parameters(), lr=self.args.lr, betas=(0.5,0.999), eps = 1e-8)
        self.g2_optim = torch.optim.Adam(self.g2.parameters(), lr=self.args.lr, betas=(0.5,0.999), eps = 1e-8)
        
        self.sup_loss = nn.L1Loss().cuda(self.args.gpu) if self.args.sup_loss=='l1' else nn.MSELoss().cuda(self.args.gpu)

        self.PSNR = PSNR(self.args.gpu, ycbcr=False)
        self.lpips = lpips.LPIPS(net='alex',version="0.1").cuda(self.args.gpu)
        self.ensemble_weight = torch.nn.Parameter(torch.tensor(0.5)).cuda(self.args.gpu)

        print("Train_source_1 path : ", self.args.train_s_path1)
        print("Train_source_2 path : ", self.args.train_s_path2)
        print("Train_source_label path : ", self.args.train_label_path)
        print("Test_X path : ", self.args.test_x_path)
        print("Test_Y path : ", self.args.test_y_path)

        print("Train_source_1, train_source_2, train_label length : ", len(self.dataset_tr))
        print("Train_target length : ", len(self.dataset_tr_target))
        print("Test_X, test_Y length : ", len(self.dataset_test))
    
    def load(self):
        g1_state = torch.load(self.args.g1_ckpt_path, map_location='cuda:'+str(self.args.gpu))
        g2_state = torch.load(self.args.g2_ckpt_path, map_location='cuda:'+str(self.args.gpu))


        if self.args.backbone=='rrdb':
            g1_state = g1_state['params']
            g2_state = g2_state['params']
        elif self.args.backbone == 'edsr':
            if list(g1_state.keys())[0]=='EDSR':
                g1_state = g1_state['EDSR']
            if list(g2_state.keys())[0]=='EDSR':
                g2_state = g2_state['EDSR']

        self.g1.load_state_dict(g1_state)
        self.g2.load_state_dict(g2_state)
        
        self.model_g2.eval()
        checkpoint = torch.load(self.args.checkpoint, map_location='cuda:' + str(self.args.gpu))
        self.model_g2.load_state_dict(checkpoint['model_g_state_dict'])

        self.model_g1.eval()
        checkpoint = torch.load(self.args2.checkpoint, map_location='cuda:' + str(self.args.gpu))
        self.model_g1.load_state_dict(checkpoint['model_g_state_dict'])

        
        print("CKPT loaded")

    def train(self):
        
        print("Train")
        loss_dict = {}
        val_loss_dict = {}
        check_folder(os.path.join(self.args.log_path,"ckpt"))
        for iteration in range(self.args.iterations):
            if iteration%self.args.decay_step==0:
                n = iteration//self.args.decay_step
                self.g1_optim.param_groups[0]['lr'] = self.args.lr/2**n
                self.g2_optim.param_groups[0]['lr'] = self.args.lr/2**n
                            
            try:
                img_y, fname = train_iter.next()
                    
            except:
                train_iter = iter(self.loader_tr)
                img_y, fname = train_iter.next()

            img_y = img_y.cuda(self.args.gpu)

            try:
                img_t, fname_t = train_target_iter.next()
            except:
                train_target_iter = iter(self.loader_tr_target)
                img_t, fname_t = train_target_iter.next()
            img_t = img_t.cuda(self.args.gpu)

            # make img2
            with torch.no_grad():
                img1 = self.model_g1(img_y)
                img2 = self.model_g2(img_y)


            
            # Update g1
            if self.args.kd_feat:
                fake_g1_img1, feat_g1_img1 = self.g1(img1, return_feature=True)
                fake_g1_img1 = torch.clamp(fake_g1_img1,min=0,max=1)

                fake_g1_img2, feat_g1_img2 = self.g1(img2, return_feature=True)
                fake_g1_img2 = torch.clamp(fake_g1_img2,min=0,max=1)

                fake_g2_img2, feat_g2_img2 = self.g2(img2, return_feature=True)
                fake_g2_img2 = torch.clamp(fake_g2_img2,min=0,max=1)

                g1_supervised_l1loss = self.sup_loss(fake_g1_img1, img_y)
                g1_l1loss_img2 = self.sup_loss(fake_g1_img2, fake_g2_img2.detach())
                g1_feat_loss = self.sup_loss(feat_g1_img2, feat_g2_img2.detach())
                if iteration>0:
                    loss_dict['g1_feat_loss'] = g1_feat_loss
            else:
                fake_g1_img1 = self.g1(img1)
                fake_g1_img1 = torch.clamp(fake_g1_img1,min=0,max=1)

                fake_g1_img2 = self.g1(img2)
                fake_g1_img2 = torch.clamp(fake_g1_img2,min=0,max=1)

                fake_g2_img2 = self.g2(img2)
                fake_g2_img2 = torch.clamp(fake_g2_img2,min=0,max=1)

                g1_supervised_l1loss = self.sup_loss(fake_g1_img1, img_y)
                g1_l1loss_img2 = self.sup_loss(fake_g1_img2, fake_g2_img2.detach())

            if iteration>0:
                loss_dict["g1_img1_imgy_psnr"] = self.PSNR(fake_g1_img1, img_y)
                loss_dict["g1_img2_g2_psnr"] = self.PSNR(fake_g1_img2, fake_g2_img2)
                loss_dict["g1_img2_imgy_psnr"] = self.PSNR(fake_g1_img2, img_y)
            else:
                loss_dict["g1_img1_imgy_psnr"] = 0
                loss_dict["g1_img2_g2_psnr"] = 0
                loss_dict["g1_img2_imgy_psnr"] = 0

            g1_loss = self.args.sup_weight * g1_supervised_l1loss + self.args.col_weight * g1_l1loss_img2 * (self.args.col_start_iter < iteration)
            if self.args.kd_feat:
                g1_loss += self.args.feat_weight*g1_feat_loss

            g1_loss.backward(retain_graph=True)
            self.g1_optim.step()
            self.g1_optim.zero_grad()
            
            # Update g2
            # fake_g1_img1 = self.g1(img1)
            # fake_g2_img2 = self.g2(img2)

            if self.args.kd_feat:
                fake_g2_img1, feat_g2_img1 = self.g2(img1, return_feature=True)
                g2_feat_loss = self.sup_loss(feat_g2_img1, feat_g1_img1.detach())
                if iteration>0:
                    loss_dict['g2_feat_loss'] = g2_feat_loss
                
            else:
                fake_g2_img1 = self.g2(img1)
            fake_g2_img1 = torch.clamp(fake_g2_img1,min=0,max=1)

            g2_supervised_l1loss = self.sup_loss(fake_g2_img2, img_y)
            g2_l1loss_img1 = self.sup_loss(fake_g2_img1, fake_g1_img1.detach())
            
            if iteration>0:
                loss_dict["g2_img2_imgy_psnr"] = self.PSNR(fake_g2_img2, img_y)
                loss_dict["g2_img1_g1_psnr"] = self.PSNR(fake_g2_img1, fake_g1_img1)
                loss_dict["g2_img1_imgy_psnr"] = self.PSNR(fake_g2_img1, img_y)
            else:
                loss_dict["g2_img2_imgy_psnr"] = 0
                loss_dict["g2_img1_g1_psnr"] = 0
                loss_dict["g2_img1_imgy_psnr"] = 0

            g2_loss = self.args.sup_weight * g2_supervised_l1loss + self.args.col_weight * g2_l1loss_img1 * (self.args.col_start_iter < iteration)
            if self.args.kd_feat:
                g2_loss += self.args.feat_weight*g2_feat_loss
            g2_loss.backward()
            self.g2_optim.step()
            self.g2_optim.zero_grad()

            fake_g1_img_t = self.g1(img_t)
            fake_g1_img_t = torch.clamp(fake_g1_img_t,min=0,max=1)

            fake_g2_img_t = self.g2(img_t)
            fake_g2_img_t = torch.clamp(fake_g2_img_t,min=0,max=1)
            
            with torch.no_grad():
                pseudo_hr = 0.5 * fake_g1_img_t + 0.5*fake_g2_img_t
            
            g1_adapt_l1_loss = (self.args.col_start_iter < iteration) * self.sup_loss(pseudo_hr, fake_g1_img_t) * iteration/self.args.iterations
            g1_adapt_l1_loss.backward()
            self.g1_optim.step()
            self.g1_optim.zero_grad()

            g2_adapt_l1_loss = (self.args.col_start_iter < iteration)* self.sup_loss(pseudo_hr, fake_g2_img_t) * iteration/self.args.iterations
            g2_adapt_l1_loss.backward()
            self.g2_optim.step()
            self.g2_optim.zero_grad()

            loss_dict["g1_adapt_sup_loss"] = g1_adapt_l1_loss
            loss_dict["g2_adapt_sup_loss"] = g2_adapt_l1_loss
            
            if self.args.kd_feat:
                del g1_feat_loss
                del g2_feat_loss


                    

            print("iter: ",iteration, loss_dict)

            
            if iteration%self.args.val_freq==0 and iteration>148:
                dicts = {}
                dicts['params'] = self.g1.state_dict()
                torch.save(dicts, os.path.join(self.args.log_path, 'ckpt', f"{iteration}g1.ckpt"))

                dicts = {}
                dicts['params'] = self.g2.state_dict()
                torch.save(dicts, os.path.join(self.args.log_path, 'ckpt', f"{iteration}g2.ckpt"))
                
                with torch.no_grad():
                    psnr_val_g1 = 0
                    psnr_val_g2 = 0
                    psnr_val_ensemble = 0
                    lpips_ensemble = 0
                    save_toggle = 0
                    for test_x, test_y, fname in self.loader_test:
                        test_x, test_y = test_x.cuda(self.args.gpu), test_y.cuda(self.args.gpu)
                        fname = fname[0]
                        g1_testx = self.g1(test_x)
                        g1_testx = torch.clamp(g1_testx,min=0,max=1)
                        
                        g2_testx = self.g2(test_x)
                        g2_testx = torch.clamp(g2_testx,min=0,max=1)


                        ensemble = self.ensemble_weight*g1_testx + (1-self.ensemble_weight)*g2_testx

                        psnr_val_g1 += self.PSNR(g1_testx, test_y)/len(self.loader_test)
                        psnr_val_g2 += self.PSNR(g2_testx, test_y)/len(self.loader_test)
                        psnr_val_ensemble += self.PSNR(ensemble, test_y)/len(self.loader_test)
                        lpips_ensemble += self.lpips.forward(ensemble, test_y)/len(self.loader_test)

                        if save_toggle<15:
                            tensor_imsave(g1_testx[0], os.path.join(self.args.log_path, "val", f"{iteration}"),f"g1_{fname}")
                            tensor_imsave(g2_testx[0], os.path.join(self.args.log_path, "val", f"{iteration}"),f"g2_{fname}")
                            tensor_imsave(ensemble[0], os.path.join(self.args.log_path, "val", f"{iteration}"),f"ensemble_{fname}")
                            save_toggle+=1
                        
                    val_loss_dict["psnr_val_g1"] = psnr_val_g1
                    val_loss_dict["psnr_val_g2"] = psnr_val_g2
                    val_loss_dict["psnr_val_ensemble"] = psnr_val_ensemble
                    val_loss_dict["lpips"] = lpips_ensemble
                        
            

                print(val_loss_dict)
            
                

    def inference(self):# SR inference for whole data
        with torch.no_grad():
            for test_x, test_y, fname in self.loader_test:
                test_x, test_y = test_x.cuda(self.args.gpu), test_y.cuda(self.args.gpu)
                fname = fname[0]
                g1_testx = self.g1(test_x)
                g2_testx = self.g2(test_x)
                ensemble = self.ensemble_weight*g1_testx + (1-self.ensemble_weight)*g2_testx

                
                tensor_imsave(g1_testx[0], os.path.join(self.args.inference_path, "val"),f"g1_{fname}")
                tensor_imsave(g2_testx[0], os.path.join(self.args.inference_path, "val"),f"g2_{fname}")
                tensor_imsave(ensemble[0], os.path.join(self.args.inference_path, "val"),f"ensemble_{fname}")
                    
                
