import torch.utils.data as data
import random
from PIL import Image
from torchvision import transforms
import os
import os.path
from utils import *
from transforms import *
import torchvision.transforms.functional as TF
import random
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir, extensions, label):
    # Returns the tuples (path, label) of data
    # Now, label is same with fname
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                # path = os.path.join(root, fname)
                item = (root, fname)
                images.append(item)
        if root==dir:
            break
        else:
            NotImplementedError("Dataloader")

    return images

class TwoSourceDatasetFolder(data.Dataset):
    def __init__(self, root1, root2, root_y, label, extensions = ['.jpg', '.png', '.jpeg', '.bmp'], transform=None, return_two_img = False, lrsize = 32, scale_factor = 4, random = True):
        # root -- Dataset folder path
        # label -- 0 / 1
        # extensions -- list of alowed extensions
        # transform -- Transform applied to image
        # return_two_img -- Whether to return both big and small imgs or not
        # big_imsize -- Size of bigger img. Matters only if return_two_img is True
        # scale_factor -- SR scale factor. Matters only if return_two_img is True
        samples1 = make_dataset(root1, extensions, label)
        samples2 = make_dataset(root2, extensions, label)
        samples_y = make_dataset(root_y, extensions, label)

        assert len(samples1)==len(samples2)==len(samples_y), f"len(samples1) = {len(samples1)} len(samples2) = {len(samples2)} len(y) = {len(samples_y)}"

        if len(samples1) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        
        self.root1 = root1
        self.root2 = root2
        self.root_y = root_y

        self.extensions = extensions
        self.samples1 = samples1
        self.samples2 = samples2
        self.samples_y = samples_y

        
        
        self.lrsize = lrsize
        self.scale_factor = scale_factor
        self.random = random
    def transform(self, imgs, img_y):
        i, j, h, w = transforms.RandomCrop.get_params(
            imgs[0], output_size=(self.lrsize, self.lrsize))
        angle = random.choice([0,90,-90,-180])
        p1 = random.random()
        p2 = random.random()
        
        for idx in range(len(imgs)):
            imgs[idx] = TF.to_tensor(TF.crop(imgs[idx], i,j,h,w))
            if self.random:
                imgs[idx] = TF.rotate(imgs[idx], angle)
                if p1>0.5:
                    imgs[idx] = TF.hflip(imgs[idx])
                if p2>0.5:
                    imgs[idx] = TF.vflip(imgs[idx])
        img_y = TF.crop(img_y, i*self.scale_factor, j*self.scale_factor, h*self.scale_factor, w*self.scale_factor, )
        if self.random:
            img_y = TF.rotate(img_y,angle)
            if p1>0.5:
                img_y = TF.hflip(img_y)
            if p2>0.5:
                img_y = TF.vflip(img_y)
        
        img_y = TF.to_tensor(img_y)
        assert img_y.size()[1] == imgs[0].size(2)*self.scale_factor

        return imgs, img_y

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        root1, fname1 = self.samples1[index]
        root2, _ = self.samples2[index]
        root_y, _ = self.samples_y[index]
        
        img1 = pil_loader(os.path.join(root1, fname1))
        img2 = pil_loader(os.path.join(root2, fname1))
        img_y = pil_loader(os.path.join(root_y, fname1))
        
        imgs, img_y = self.transform([img1,img2], img_y)


        return img_y, fname1
    
    def __len__(self):
        return len(self.samples1)

class Random90Rot(object):
    def __call__(self, img):
        angle_list = [0,-90,-180,90]
        angle = random.choice(angle_list)

        return transforms.functional.rotate(img, angle)

class DatasetFolder(data.Dataset):
    def __init__(self, root, label, extensions = ['.jpg', '.png', '.jpeg', '.bmp'], transform=None, return_two_img = False, rot=False, big_imsize = 128, scale_factor = 4):
        # root -- Dataset folder path
        # label -- 0 / 1
        # extensions -- list of alowed extensions
        # transform -- Transform applied to image
        # return_two_img -- Whether to return both big and small imgs or not
        # big_imsize -- Size of bigger img. Matters only if return_two_img is True
        # scale_factor -- SR scale factor. Matters only if return_two_img is True
        samples = make_dataset(root, extensions, label)

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        
        self.root = root
        self.extensions = extensions
        self.samples = samples
        
        self.return_two_img = return_two_img
        self.big_imsize = big_imsize
        self.scale_factor = scale_factor
        self.rot = rot
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, fname = self.samples[index]
        sample = pil_loader(os.path.join(path, fname))
        


        if self.return_two_img:
            t1 = transforms.Compose([
            Random90Rot(),
            transforms.RandomCrop((self.big_imsize,self.big_imsize))])

            t2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

            sample = t1(sample)
            sample_big = t2(sample)
            sample_small = transforms.Resize((self.big_imsize//self.scale_factor, self.big_imsize//self.scale_factor), Image.BICUBIC)(sample)
            sample_small = t2(sample_small)

            return sample_big, sample_small, target

        
        sample = self.transform(sample)
        # if self.rot:
        #     sample = TF.rotate(sample, 90)
        sample = TF.to_tensor(sample)

        return sample, fname
    
    def __len__(self):
        return len(self.samples)
    def transform(self, img):
        lr_size = self.big_imsize//self.scale_factor
        
        t = transforms.RandomCrop((lr_size,lr_size))
        return t(img)



class TestDatasetFolder(data.Dataset):
    def __init__(self, root1, root2, label, extensions = ['.jpg', '.png', '.jpeg', '.bmp'], transform=None, return_two_img = False, lrsize = 32, scale_factor = 4, random = True):
        # root -- Dataset folder path
        # label -- 0 / 1
        # extensions -- list of alowed extensions
        # transform -- Transform applied to image
        # return_two_img -- Whether to return both big and small imgs or not
        # big_imsize -- Size of bigger img. Matters only if return_two_img is True
        # scale_factor -- SR scale factor. Matters only if return_two_img is True
        samples1 = make_dataset(root1, extensions, label)
        samples2 = make_dataset(root2, extensions, label)

        assert len(samples1)==len(samples2)

        
        self.root1 = root1
        self.root2 = root2


        self.extensions = extensions
        self.samples1 = samples1
        self.samples2 = samples2
        
        self.lrsize = lrsize
        self.scale_factor = scale_factor
        self.random = random
    def transform(self, imgs, img_y):
        i, j, h, w = transforms.RandomCrop.get_params(
            imgs[0], output_size=(self.lrsize, self.lrsize))
        for idx in range(len(imgs)):
            imgs[idx] = TF.to_tensor(TF.crop(imgs[idx], i,j,h,w))
        img_y = TF.crop(img_y, i*self.scale_factor, j*self.scale_factor, h*self.scale_factor, w*self.scale_factor, )
        img_y = TF.to_tensor(img_y)
        assert img_y.size()[1] == imgs[0].size(2)*self.scale_factor

        return imgs, img_y

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        root1, fname1 = self.samples1[index]
        root2, _ = self.samples2[index]

        
        img1 = TF.to_tensor(pil_loader(os.path.join(root1, fname1)))
        img2 = TF.to_tensor(pil_loader(os.path.join(root2, fname1)))

        
        


        return img1, img2, fname1
    
    def __len__(self):
        return len(self.samples1)