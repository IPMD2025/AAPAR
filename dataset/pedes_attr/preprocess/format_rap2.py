import os
import sys
sys.path.append('/media/data2/**/clip-pa100k-0405/')
import numpy as np
import random
import pickle
from scipy.io import loadmat
from easydict import EasyDict
import glob
import cv2
from clipS import clip
import re

np.random.seed(0)
random.seed(0)

attr_words = [
    'A pedestrian with a bald head', 'A pedestrian with long hair', 'A pedestrian with black hair',
    'A pedestrian wearing a hat', 'A pedestrian wearing glasses',
    'A pedestrian in a shirt', 'A pedestrian in a sweater', 'A pedestrian in a vest',
    'A pedestrian in a t-shirt', 'A pedestrian in cotton clothing', 'A pedestrian in a jacket',
    'A pedestrian dressed in a suit', 'A pedestrian in tight-fitting clothes', 'A pedestrian in a short-sleeved top',
    'A pedestrian with other upper wear', 'A pedestrian in long trousers', 'A pedestrian in a skirt',
    'A pedestrian in a short skirt', 'A pedestrian in a dress', 'A pedestrian in jeans',
    'A pedestrian in tight trousers', 'A pedestrian in leather shoes', 'A pedestrian in sports shoes',
    'A pedestrian wearing boots', 'A pedestrian in cloth shoes', 'A pedestrian in casual shoes',
    'A pedestrian in other footwear', 'A pedestrian carrying a backpack', 'A pedestrian with a shoulder bag',
    'A pedestrian with a handbag', 'A pedestrian with a box', 'A pedestrian with a plastic bag',
    'A pedestrian with a paper bag', 'A pedestrian carrying a hand trunk', 'A pedestrian with other attachments',
    'A pedestrian under the age of 16', 'A pedestrian between the ages of 17 and 30', 'A pedestrian between the ages of 31 and 45',
    'A pedestrian between the ages of 46 and 60', 'A female pedestrian', 
    'A pedestrian with a larger body build', 'A pedestrian with a normal body build', 'A pedestrian with a slender body build',
    'A customer', 'An employee',
    'A pedestrian making a call', 'A pedestrian engaged in conversation', 'A pedestrian gathered with others',
    'A pedestrian holding something', 'A pedestrian pushing something', 'A pedestrian pulling something',
    'A pedestrian carrying something in their arm', 'A pedestrian carrying something in their hand', 'A pedestrian engaged in other actions',
]# 54 

# attr_words_1 = ['A female pedestrian', 'A pedestrian under the age of 16', 'A pedestrian between the ages of 17 and 30', 'A pedestrian between the ages of 31 and 45',
#     'A pedestrian between the ages of 46 and 60','A pedestrian with a larger body build', 'A pedestrian with a normal body build', 'A pedestrian with a slender body build',
#     'A customer', 'An employee',
#     'A pedestrian making a call', 'A pedestrian engaged in conversation', 'A pedestrian gathered with others',
#     'A pedestrian holding something', 'A pedestrian pushing something', 'A pedestrian pulling something',
#     'A pedestrian carrying something in their arm', 'A pedestrian carrying something in their hand', 'A pedestrian engaged in other actions',
    
#     'A pedestrian wearing a hat', 'A pedestrian wearing glasses',
#     'A pedestrian with a bald head', 'A pedestrian with long hair', 'A pedestrian with black hair',
    
#     'A pedestrian in a shirt', 'A pedestrian in a sweater', 'A pedestrian in a vest',
#     'A pedestrian in a t-shirt', 'A pedestrian in cotton clothing', 'A pedestrian in a jacket',
#     'A pedestrian dressed in a suit', 'A pedestrian in tight-fitting clothes', 'A pedestrian in a short-sleeved top',
#     'A pedestrian with other upper wear',
    
#      'A pedestrian in long trousers', 'A pedestrian in a skirt',
#     'A pedestrian in a short skirt', 'A pedestrian in a dress', 'A pedestrian in jeans',
#     'A pedestrian in tight trousers', 'A pedestrian in leather shoes', 'A pedestrian in sports shoes',
#     'A pedestrian wearing boots', 'A pedestrian in cloth shoes', 'A pedestrian in casual shoes',
#     'A pedestrian in other footwear', 'A pedestrian carrying a backpack', 'A pedestrian with a shoulder bag',
#     'A pedestrian with a handbag', 'A pedestrian with a box', 'A pedestrian with a plastic bag',
#     'A pedestrian with a paper bag', 'A pedestrian carrying a hand trunk', 'A pedestrian with other attachments',
     
    
# ]

group_order = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
               36, 37, 38, 39, 40, 41, 42, 43, 44, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 45, 46, 47, 48, 49, 50, 51, 52, 53]



def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def get_label_embeds(labels):
    model, preprocess = clip.load("ViT-L/14", device='cpu',download_root='/media/backup/**/pretrained/')
    text = clip.tokenize(labels)
    embeddings = model.encode_text(text)
    return embeddings


def generate_data_description(save_dir, reorder, new_split_path, version):
    data = loadmat(os.path.join(save_dir, 'RAP_annotation/RAP_annotation.mat'))
    data = data['RAP_annotation']
    dataset = EasyDict()
    dataset.description = 'rap2'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'RAP_png')

    folder_path = os.path.join(save_dir, 'RAP_dataset')
    save_path=os.path.join(save_dir, 'RAP_png')
    file_list = os.listdir(folder_path)
    image_files = glob.glob(folder_path + "/*.png")
    for image_file in image_files:
        imagename,_ =os.path.splitext(os.path.basename(image_file))
        image_save_path=save_path+"/"+imagename+".png"
        if os.path.exists(image_save_path):
                continue
        else:
            img=cv2.imread(image_file)#112*410,#1108
            heigth,width = img.shape[0],img.shape[1]   #获取图片的长和宽#255,104
            # p=max(heigth,width)
            # n=p/224
            n = heigth/224
            #等比例缩小图片
            new_heigth = heigth/n 
            new_width = width/n
            a,b,c=0,0,0
            if (int(new_heigth) <=224) and (int(new_width) <=224):
                img = cv2.resize(img, (int(new_width), int(new_heigth)))
                a = int((224 - new_heigth) / 2)
                b = int((224 - new_width) / 2)
                change_width=a*2+img.shape[0]
                if(change_width<224): 
                    c=a+224-change_width
                elif (change_width==224): 
                    c=a
                else : 
                    c=a-224+change_width
                
            cv2.imwrite(save_path+"/"+imagename+".png", cv2.copyMakeBorder(img, c,a, b, b, cv2.BORDER_CONSTANT, value=[0, 0, 0]))


    dataset.image_name = []
    for i in range(84928):
        imagename,_ =os.path.splitext(data['name'][0][0][i][0][0])
        dataset.image_name.append(imagename + '.png')
    raw_attr_name = [data['attribute'][0][0][i][0][0] for i in range(152)]
    raw_label = data['data'][0][0]
    selected_attr_idx = (data['selected_attribute'][0][0][0] - 1)[group_order].tolist()  # 54

    color_attr_idx = list(range(31, 45)) + list(range(53, 67)) + list(range(74, 88))  # 42
    extra_attr_idx = np.setdiff1d(range(152), color_attr_idx + selected_attr_idx).tolist()[:24]
    extra_attr_idx = extra_attr_idx[:15] + extra_attr_idx[16:]

    # dataset.label = raw_label[:, selected_attr_idx + color_attr_idx + extra_attr_idx]  # (n, 119)
    # dataset.attr_name = [raw_attr_name[i] for i in selected_attr_idx + color_attr_idx + extra_attr_idx]
    dataset.label = raw_label[:, selected_attr_idx]  # (n, 119)
    
    dataset.attr_name = [raw_attr_name[i] for i in selected_attr_idx]

    # words = np.array(attr_words)
    # dataset.attr_words = np.array(words[group_order])
    # dataset.attr_vectors = get_label_embeds(words[group_order])
    dataset.attr_words = np.array(attr_words)
    dataset.attr_vectors = get_label_embeds(attr_words)

    dataset.label_idx = EasyDict()
    dataset.label_idx.eval = list(range(54))  # 54
    dataset.label_idx.color = list(range(54, 96))  # not aligned with color label index in label
    dataset.label_idx.extra = list(range(96, 119))  # not aligned with extra label index in label

    
    new_label = dataset.label
    des=[]
    for k in range(len(new_label)):
        description=''
        if new_label[k][39]==1:
            description +="The "+ 'woman' 
        else:
            description +="The "+ 'man' 
        
        career=['a customer', 'an employee']  
    
        for i in range(43,45):
            if  new_label[k][i]==1:
                description += ' is '
                description += career[i-43]
                # description += ','
                      
        age=[" under the age of 16 ", " between the ages of 17 and 30 ",
             " between the ages of 31 and 45 ", " between the ages of 46 and 60 "]
        ageind=0
        for i in range(35,39):
            if  new_label[k][i]==1:
                description += age[i-35]
                ageind+=1
        # if ageind==0:
        #     description += " unknown age"
        description +=" carrying "
        carry=["a backpack", "a shoulder bag",  "a handbag",  "a box","a plastic bag","a paper bag","a hand trunk","other attachments"]
        carryind=0
        for i in range(27,35):
            if  new_label[k][i]==1:
                if carryind >= 1:
                    description += ", "
                description += carry[i-27]
                carryind+=1
        if carryind==0:
            # description += ", "
            description +=" nothing"
        
        hair = ["a bald head","long hair","black hair"] 
        hairind = 0   
        description +=" with " 
        for i in range(3): 
            if new_label[k][i]==1: 
                if hairind >= 1:
                    description += ", "
                description += hair[i]  
                hairind += 1
        # if new_label[k][0]!=1 and hairind!=0:
        #     description +=" hair, "
        
        headwear=[ "a hat", "glasses"]
        headind=0
        for i in range(3,5): 
            if  new_label[k][i]==1:
                description += ", "
                # if headind >= 1:
                #     description += ", "
                description += headwear[i-3]
                headind+=1     
        # if headind==0:
        #     description += ", "
        #     description +="no headwear"
        
        body = ['a larger body build', 'a normal body build', 'a slender body build']
        bodyind=0
        for i in range(40,43):
            if  new_label[k][i]==1:
                # if bodyind > 0:
                description += ", "
                description += body[i-40]
                bodyind+=1
        # if bodyind==0:
        #     description +="unknown body build"
                        
        
       
        
                    
        upercloth=['a shirt', 'a sweater', 'a vest','t-shirt', 'cotton clothes', 'a jacket','a suit','tight-fitting upper clothes', 'short sleeves','other upper wear']
        uperwear=0
        for i in range(5,15):
            if  new_label[k][i]==1:
                description +=", "
                # if uperwear != 0:
                #     description += ', '
                description += upercloth[i-5]
                uperwear+=1
        # if uperwear==0:
        #     description += ", "
        #     description += "unknown upperwear" 
        
        
        #lowbody    
            
        lowercloth=['long trousers', 'a skirt','a short skirt', 'a dress', 'jeans','tight trousers']
        lowerwear=0
        for i in range(15,21):
            if  new_label[k][i]==1:
                description += ", "
                # if lowerwear != 0:
                #     description += ', '
                description += lowercloth[i-15]
                lowerwear+=1
        # if lowerwear==0:
        #     description += ", "
        #     description +=" other lower wear" 
        
        
        #footwear
        footwear=['leather shoes', 'sports shoes','boots', 'cloth shoes', 'casual shoes','other foot wear']
        footind=0
        for i in range(21,27):
            if  new_label[k][i]==1:
                description += ", "
                description += footwear[i-21]
                footind+=1
        # if footind==0:
        #     description += ", "
        #     description +="unknown shoes"
        
        # description += ", "   
        
                
            
        action=['making a call', 'engaging in conversation', 'gathering with others','holding something', 'pushing something', 'pulling something',
        'carrying something in their arm', 'carrying something in their hand', 'engaging in other actions']
        actionind =0
        for i in range(45,54):
            if  new_label[k][i]==1:
                # if actionind==0:
                description += ", " 
                # else:
                #     description += ', '
                description += action[i-45]
                actionind += 1
        # if actionind==0:
        #     description += "engaging in unknown actions"
        
        description += "."
        description = pre_caption(description,77)    
        des.append(description)
    dataset.des=des
    
    
    
    if reorder:
        dataset.label_idx.eval = list(range(54))

    dataset.partition = EasyDict()
    dataset.partition.train = []
    dataset.partition.val = []
    dataset.partition.test = []
    dataset.partition.trainval = []

    dataset.weight_train = []
    dataset.weight_trainval = []

    if new_split_path:

        # remove Age46-60
        dataset.label_idx.eval.remove(38)  # 54

        with open(new_split_path, 'rb+') as f:
            new_split = pickle.load(f)

        train = np.array(new_split.train_idx)
        val = np.array(new_split.val_idx)
        test = np.array(new_split.test_idx)
        trainval = np.concatenate((train, val), axis=0)

        print(np.concatenate([trainval, test]).shape)

        dataset.partition.train = train
        dataset.partition.val = val
        dataset.partition.trainval = trainval
        dataset.partition.test = test

        weight_train = np.mean(dataset.label[train], axis=0).astype(np.float32)
        weight_trainval = np.mean(dataset.label[trainval], axis=0).astype(np.float32)

        print(weight_trainval[38])

        dataset.weight_train.append(weight_train)
        dataset.weight_trainval.append(weight_trainval)
        with open(os.path.join(save_dir, f'dataset_zs_run{version}.pkl'), 'wb+') as f:
            pickle.dump(dataset, f)

    else:
        for idx in range(5):
            train = data['partition_attribute'][0][0][0][idx]['train_index'][0][0][0] - 1
            val = data['partition_attribute'][0][0][0][idx]['val_index'][0][0][0] - 1
            test = data['partition_attribute'][0][0][0][idx]['test_index'][0][0][0] - 1
            trainval = np.concatenate([train, val])
            dataset.partition.train.append(train)
            dataset.partition.val.append(val)
            dataset.partition.test.append(test)
            dataset.partition.trainval.append(trainval)
            # cls_weight
            weight_train = np.mean(dataset.label[train], axis=0)
            weight_trainval = np.mean(dataset.label[trainval], axis=0)
            dataset.weight_train.append(weight_train)
            dataset.weight_trainval.append(weight_trainval)
        with open(os.path.join(save_dir, 'dataset_all.pkl'), 'wb+') as f:
            pickle.dump(dataset, f)

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption

if __name__ == "__main__":
    save_dir = '/media/data2/**/dataset/RAP2/'
    generate_data_description(save_dir,False,False,2)
    # reorder = True

    # for i in range(5):
    #     new_split_path = f'/mnt/data1/jiajian/code/Rethinking_of_PAR/datasets/jian_split/index_rap2_split_id50_img300_ratio0.03_{i}.pkl'
    #     generate_data_description(save_dir, reorder, new_split_path, i)
