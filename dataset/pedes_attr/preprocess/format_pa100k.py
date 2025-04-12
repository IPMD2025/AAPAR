import os
import sys
sys.path.append('/media/data2/**/clip-pa100k-0505')
import numpy as np
import random
import pickle
import glob
import cv2
from clipS import clip
import re


from easydict import EasyDict
from scipy.io import loadmat
from sentence_transformers import SentenceTransformer

np.random.seed(0)
random.seed(0)

group_order = [7, 8, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 9, 10, 11, 12, 1, 2, 3, 0, 4, 5, 6]
# attr_words = [
#     'female',
#     'age over 60', 'age 18 to 60', 'age less 18',
#     'front', 'side', 'back',
#     'hat', 'glasses', 
#     'hand bag', 'shoulder bag', 'backpack', 'hold objects in front', 
#     'short sleeve', 'long sleeve', 'upper stride', 'upper logo', 'upper plaid', 'upper splice',
#     'lower stripe', 'lower pattern', 'long coat', 'trousers', 'shorts', 'skirt and dress', 'boots'
# ]

attr_words = [
    'A female pedestrian',
    'A pedestrian over the age of 60', 'A pedestrian between the ages of 18 and 60', 'A pedestrian under the age of 18',
    'A pedestrian seen from the front', 'A pedestrian seen from the side', 'A pedestrian seen from the back',
    'A pedestrian wearing a hat', 'A pedestrian wearing glasses',
    'A pedestrian with a handbag', 'A pedestrian with a shoulder bag', 'A pedestrian with a backpack', 'A pedestrian holding objects in front',
    'A pedestrian in short-sleeved upper wear', 'A pedestrian in long-sleeved upper wear', 'A pedestrian in stride upper wear', 'A pedestrian in upper wear with a logo', 'A pedestrian in plaid upper wear', 'A pedestrian in splice upper wear',
    'A pedestrian in striped lower wear', 'A pedestrian in patterned lower wear', 'A pedestrian in a long coat', 'A pedestrian in trousers', 'A pedestrian in shorts', 'A pedestrian in skirts and dresses', 'A pedestrian wearing boots'
]
index =range(26)#[0,3,5,7,11,14,22,25]# 
def get_label_embeds1(labels):
    model = SentenceTransformer('/media/sdb/**/pretrained/all-mpnet-base-v2')
    embeddings = model.encode(labels)
    return embeddings

def get_label_embeds(labels):
    model, preprocess = clip.load("ViT-L/14", device='cpu',download_root='/media/backup/**/pretrained/')
    text = clip.tokenize(labels)
    embeddings = model.encode_text(text)
    return embeddings

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def generate_data_description(save_dir, reorder):
    """
    create a dataset description file, which consists of images, labels
    """
    # pa100k_data = loadmat('/mnt/data1/jiajian/dataset/attribute/PA100k/annotation.mat')
    pa100k_data = loadmat(os.path.join(save_dir, 'annotation.mat'))

    dataset = EasyDict()
    dataset.description = 'pa100k'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'Pa100k_datasets')#data

    folder_path = os.path.join(save_dir, 'data')
    save_path=os.path.join(save_dir, 'Pa100k_datasets')
    file_list = os.listdir(folder_path)
    image_files = glob.glob(folder_path + "/*.jpg")
    for image_file in image_files:
        imagename,_ =os.path.splitext(os.path.basename(image_file))
        image_save_path=save_path+"/"+imagename+".jpg"
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
                
            cv2.imwrite(save_path+"/"+imagename+".jpg", cv2.copyMakeBorder(img, c,a, b, b, cv2.BORDER_CONSTANT, value=[0, 0, 0]))



    train_image_name = [pa100k_data['train_images_name'][i][0][0] for i in range(80000)]
    val_image_name = [pa100k_data['val_images_name'][i][0][0] for i in range(10000)]
    test_image_name = [pa100k_data['test_images_name'][i][0][0] for i in range(10000)]
    dataset.image_name = train_image_name + val_image_name + test_image_name

    dataset.label = np.concatenate((pa100k_data['train_label'], pa100k_data['val_label'], pa100k_data['test_label']), axis=0)
    assert dataset.label.shape == (100000, 26)
    label_arrange=dataset.label[:,group_order]
    raw_lable=dataset.label
    dataset.label=label_arrange

    dataset.attr_name = [pa100k_data['attributes'][i][0][0] for i in range(26)]
    words = np.array(attr_words)
    dataset.attr_words = np.array(words[group_order])
    dataset.attr_vectors = get_label_embeds(words[group_order])
    # dataset.attr_words = np.array(attr_words)
    # dataset.attr_vectors = get_label_embeds(attr_words)

    # dataset.label_idx = EasyDict()
    # dataset.label_idx.eval = list(range(26))

    # if reorder:
    #     dataset.label_idx.eval = group_order
    #gen description
    des=[]
    for k in range(len(raw_lable)):
        for j in range(len(index)):
            if raw_lable[k][index[j]]==0:
                raw_lable[k][index[j]] = 1
            else:
                raw_lable[k][index[j]] = 0
        description=''
        orient = ['From the front view', 'From the side view', 'From the back view']
        orientind=0
        for i in range(4,7):
            if  raw_lable[k][i]==1:
                if orientind !=0:
                    description += ', '
                description += orient[i-4]
                orientind+=1
        if orientind==0:
            description +="From the unknown view" 
        
        description += ", "
        if raw_lable[k][0]==1:
            description +="the "+ 'woman' 
        else:
            description +="the "+ 'man' 
               
        description +=" is"       
        age=[' over the age of 60', ' between the ages of 18 and 60', ' under the age of 18']
        ageind=0
        for i in range(1,4):
            if  raw_lable[k][i]==1:
                if ageind!= 0:
                    description += ', '
                description += age[i-1]
                ageind+=1
        if ageind==0:
            description += " unknown age"
            
        description +=", carrying"     
        
        carry=[' the hand bag', ' the shoulder bag', ' the backpack', ' the objects in front']
        carryind=0
        for i in range(9,13):
            if  raw_lable[k][i]==1:
                if carryind != 0:
                    description += ', '
                description += carry[i-9]
                carryind+=1
        if carryind==0:
            description +=" nothing"
            # description +=carry[3]
          
        description += ", with"
        
        upercloth=[' the short sleeve', ' the long sleeve', ' the upper stride', ' the upper logo', ' the upper plaid', ' the upper splice']
        uperwear=0
        for i in range(13,19):
            if  raw_lable[k][i]==1:
                if uperwear != 0:
                    description += ','
                description += upercloth[i-13]
                uperwear+=1
        if uperwear==0:
            description += " unknown upperwear" 
        description += ","
        
        lowercloth=[' the lower stripe', ' the lower pattern', ' the long coat', ' the trousers', ' the shorts', ' the skirt and dress',]
        lowerwear=0
        for i in range(19,25):
            if  raw_lable[k][i]==1:
                if lowerwear != 0:
                    description += ','
                description += lowercloth[i-19]
                lowerwear+=1
        if lowerwear==0:
            description +=" unknown lowerwear" 
            
        description +=", "
        headwear=[ 'the hat', 'the glasses']
        headind=0
        for i in range(7,9):
            if  raw_lable[k][i]==1:
                if headind != 0:
                    description += ', '
                description += headwear[i-7]
                headind+=1
        if headind==0:
            description +="no headwear"
        
        if raw_lable[k][25]==1:
            description +=", boots"
        else:
            description +=", no boots"
            
        description += "."
        description = pre_caption(description,77)    
        des.append(description)
    dataset.des=des
                

    dataset.partition = EasyDict()
    dataset.partition.train = np.arange(0, 80000)  # np.array(range(80000))
    dataset.partition.val = np.arange(80000, 90000)  # np.array(range(80000, 90000))
    dataset.partition.test = np.arange(90000, 100000)  # np.array(range(90000, 100000))
    dataset.partition.trainval = np.arange(0, 90000)  # np.array(range(90000))

    dataset.weight_train = np.mean(dataset.label[dataset.partition.train], axis=0).astype(np.float32)
    dataset.weight_trainval = np.mean(dataset.label[dataset.partition.trainval], axis=0).astype(np.float32)

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
    save_dir = '/media/data2/**/dataset/PA100K/'
    # save_dir = './data/PA100k/'
    reoder = True
    generate_data_description(save_dir, reorder=True)
