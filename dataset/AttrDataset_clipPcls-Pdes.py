import os
import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image

from tools.function import get_pkl_rootpath
import torchvision.transforms as T


class MultiModalAttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None):

        assert args.dataset in ['PA100k', 'RAP','PETA'], \
            f'dataset name {args.dataset} is not exist'

        data_path = get_pkl_rootpath(args.dataset)

        dataset_info = pickle.load(open(data_path, 'rb+'))

        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = args.dataset
        self.transform = transform
        self.target_transform = target_transform

        self.root_path = dataset_info.root

        self.attr_id = dataset_info.attr_name
        self.attr_num = len(self.attr_id)

        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]

        self.label = attr_label[self.img_idx]
        self.label_all = self.label
        
        self.label_vector = dataset_info.attr_vectors
        self.label_word = dataset_info.attr_words

        self.words = self.label_word.tolist()
        
        self.des = np.array(dataset_info.des)[self.img_idx]
        des_label=[]
        des_refine=[]
        lind=0
        for d in self.des:
            if d not in des_refine:
                des_refine.append(d)
                dlabel = lind
                lind += 1
            else:
                dlabel = des_refine.index(d)
            des_label.append(dlabel)
        self.des_label = np.array(des_label)
        # self.des_non = np.array(des_refine)
        self.des_num = len(list(set(self.des_label)))
        self.des_non= des_refine
    
        self.des_u = np.array(dataset_info.des_upper)[self.img_idx]
        des_label_u=[]
        des_refine_u=[]
        lind=0
        for d in self.des_u:
            if d not in des_refine_u:
                des_refine_u.append(d)
                dlabel_u = lind
                lind += 1
            else:
                dlabel_u = des_refine_u.index(d)
            des_label_u.append(dlabel_u)
        self.des_label_u = np.array(des_label_u)
        self.des_num_u = len(list(set(self.des_label_u)))
        self.des_non_u= des_refine_u
        
        self.des_l = np.array(dataset_info.des_lower)[self.img_idx]
        des_label_l=[]
        des_refine_l=[]
        lind=0
        for d in self.des_l:
            if d not in des_refine_l:
                des_refine_l.append(d)
                dlabel_l = lind
                lind += 1
            else:
                dlabel_l = des_refine_l.index(d)
            des_label_l.append(dlabel_l)
        self.des_label_l = np.array(des_label_l)
        self.des_num_l = len(list(set(self.des_label_l)))
        self.des_non_l= des_refine_l
        
        self.des_b = np.array(dataset_info.des_body)[self.img_idx]
        des_label_b=[]
        des_refine_b=[]
        lind=0
        for d in self.des_b:
            if d not in des_refine_b:
                des_refine_b.append(d)
                dlabel_b = lind
                lind += 1
            else:
                dlabel_b = des_refine_b.index(d)
            des_label_b.append(dlabel_b)
        self.des_label_b = np.array(des_label_b)
        self.des_num_b = len(list(set(self.des_label_b)))
        self.des_non_b= des_refine_b
        
        self.des_g = np.array(dataset_info.des_global)[self.img_idx]
        des_label_g=[]
        des_refine_g=[]
        lind=0
        for d in self.des_g:
            if d not in des_refine_g:
                des_refine_g.append(d)
                dlabel_g = lind
                lind += 1
            else:
                dlabel_g = des_refine_g.index(d)
            des_label_g.append(dlabel_g)
        self.des_label_g = np.array(des_label_g)
        self.des_num_g = len(list(set(self.des_label_g)))
        self.des_non_g= des_refine_g
        # self.des_label = np.array(dataset_info.des_label)[self.img_idx]
        
        
        # self.group_target=np.array(dataset_info.group_target)[self.img_idx]
        # self.description=[dataset_info.description[i] for i in self.img_idx]

    def __getitem__(self, index):
        imgname, gt_label, imgidx, des, des_label,des_u, des_label_u, des_l, des_label_l, des_b, des_label_b, des_g, des_label_g = self.img_id[index],self.label[index], self.img_idx[index], self.des[index], self.des_label[index], self.des_u[index], self.des_label_u[index],self.des_l[index], self.des_label_l[index], self.des_b[index], self.des_label_b[index], self.des_g[index], self.des_label_g[index]
        # group_target,description = self.group_target[index],self.description[index]
        # descrip="".join(description)#[:8]
        imgpath = os.path.join(self.root_path, imgname)
        img = Image.open(imgpath)#.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        gt_label = gt_label.astype(np.float32)

        if self.target_transform is not None:
            gt_label = self.transform(gt_label)

        label_v = self.label_vector.detach()
        #self.label_vector.astype(np.float32)
        return img, gt_label, imgname, label_v,des,des_label,des_u,des_label_u,des_l,des_label_l,des_b,des_label_b,des_g,des_label_g

    def __len__(self):
        return len(self.img_id)

def get_transform(args):
    height = args.height
    width = args.width
    normalize = T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])#mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    train_transform = T.Compose([
        # T.Lambda(fix_img),
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    valid_transform = T.Compose([
        # T.Lambda(fix_img),
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform

def fix_img(img):
    return img.convert('RGB') if img.mode != 'RGB' else img