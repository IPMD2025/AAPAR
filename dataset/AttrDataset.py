import os
import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image

from tools.function import get_pkl_rootpath
import torchvision.transforms as T


class MultiModalAttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None):

        assert args.dataset in ['PA100k', 'RAP','PETA','RAPV2'], \
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
            
        delerror=[]
        errcount=0
        for i in range(len(self.img_idx)):
            count = 0
            count = (attr_label[self.img_idx[i]] == 2).sum()
            if count==0:
                delerror.append(self.img_idx[i])
            else:
                errcount += 1
        self.img_idx = np.array(delerror)
        
        # count = (attr_label == -1).sum()#[self.img_idx[i]]
        # for k in range(len(attr_label)):
        #     for i in range(len(attr_label[k])):
        #         if attr_label[k][i]!=1 and attr_label[k][i]!=0:
        #             print(k)
        self.label = attr_label[self.img_idx]        
        self.label_all = self.label
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]

        
        
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
        self.des_non = np.array(des_refine)

        self.lablenum = len(list(set([tuple(lb) for lb in self.label_all.tolist()])))
        
        # self.des_label = np.array(dataset_info.des_label)[self.img_idx]
        self.des_num = len(list(set(self.des_label)))
        
        self.des_non= des_refine
        
        # self.group_target=np.array(dataset_info.group_target)[self.img_idx]
        # self.description=[dataset_info.description[i] for i in self.img_idx]

    def __getitem__(self, index):
        imgname, gt_label, imgidx, des, des_label = self.img_id[index], self.label[index], self.img_idx[index], self.des[index], self.des_label[index]
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
        return img, gt_label, imgname, label_v,des,des_label

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