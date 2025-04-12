import os
import sys
sys.path.append('/media/data2/**/clip-pa100k-1011-ablation-1/')
import numpy as np
import random
import pickle
import clip
import glob
import cv2
import re

from easydict import EasyDict
from scipy.io import loadmat
from sentence_transformers import SentenceTransformer
from models.pre_peta_random import petabaseDataset

np.random.seed(0)
random.seed(0)

# note: ref by annotation.md

group_order = [10, 18, 19, 30, 15, 7, 9, 11, 14, 21, 26, 29, 32, 33, 34, 6, 8, 12, 25, 27, 31, 13, 23, 24, 28, 4, 5,
               17, 20, 22, 0, 1, 2, 3, 16]
# attr_words = ["between fifteen and thirty"0, "between thirty and forty-five"1, 
#         "between forty-five and sixty"2, "Larger sixty"3,'Backpack'4, 'Other'5,
#        'Casual'6, 'Casual'7,'Formal'8,'Formal'9,'Hat'10,'Jacket'11,'Jeans'12,'LeatherShoes'13,
#        'Logo'14,'Long'15,'Male'16,'MessengerBag'17, 'Muffler'18, 
#        'Nothing'19,'Nothing'20,'Plaid'21,'PlasticBags'22,'Sandals'23, 'Shoes'24,
#        'Shorts'25,'ShortSleeve'26,'ShortSkirt'27,'Sneaker'28,'ThinStripes'29,
#        'Sunglasses'30,'Trousers'31, 'Tshirt'32,'Other'33, 'VNeck'34,]
attr_words = [
    'A pedestrian wearing a hat', 'A pedestrian wearing a muffler', 'A pedestrian with no headwear', 'A pedestrian wearing sunglasses', 'A pedestrian with long hair',
    'A pedestrian in casual upper wear', 'A pedestrian in formal upper wear', 'A pedestrian in a jacket', 'A pedestrian in upper wear with a logo', 'A pedestrian in plaid upper wear',
    'A pedestrian in a short-sleeved top', 'A pedestrian in upper wear with thin stripes', 'A pedestrian in a t-shirt', 'A pedestrian in other upper wear', 'A pedestrian in upper wear with a V-neck',
    'A pedestrian in casual lower wear', 'A pedestrian in formal lower wear', 'A pedestrian in jeans', 'A pedestrian in shorts', 'A pedestrian in a short skirt', 'A pedestrian in trousers',
    'A pedestrian in leather shoes', 'A pedestrian in sandals', 'A pedestrian in other types of shoes', 'A pedestrian in sneakers',
    'A pedestrian with a backpack', 'A pedestrian with other types of attachments', 'A pedestrian with a messenger bag', 'A pedestrian with no attachments', 'A pedestrian with plastic bags',
    'A pedestrian under the age of 30', 'A pedestrian between the ages of 30 and 45', 'A pedestrian between the ages of 45 and 60', 'A pedestrian over the age of 60',
    'A male pedestrian'
]
index= [0,4,10,18,24,25,31]
# attr_words=['This person is accessorying Hat','This person is accessorying Muffler','This person is accessorying Nothing','This person is accessorying Sunglasses','This person has Long hair',
#     'This person is wearing Casual in upper body','This person is wearing Formal in upper body','This person is wearing Jacket in upper body','This person is wearing Logo in upper body','This person is wearing Plaid in upper body',
#     'This person is wearing ShortSleeve in upper body','This person is wearing ThinStripes in upper body','This person is wearing Tshirt in upper body','This person is  wearing Other in upper body','This person is wearing VNeck in upper body',
#     'This person is wearing Casual in lower body','This person is wearing Formal in lower body','This person is wearing Jeans in lower body','This person is wearing Shorts in lower body', 'This person is wearing ShortSkirt in lower body','This person is wearing Trousers in lower body',
#     'This person is wearing LeatherShoes in foot','This person is wearing Sandals in foot','This person is wearing Shoes in foot', 'This person is wearing Sneaker in foot',
#     'This person is carrying Backpack','This person is carrying Other','This person is carrying MessengerBag','This person is carrying Nothing','This person is carrying PlasticBags',
#     'The age of this person is between fifteen and thirty years old','The age of this person is between thirty and forty-five years old','The age of this person is between forty-five and sixty years old', 'The age of this person is Larger sixty years old',
#     'This person is male']

part_words=[[0,1,2,3,16],[10,18,19,30,15],[4,5,20,22,17,7,9,11,14,21,26,29,32,33,34],[6,8,12,25,27,31,13,23,24,28]]
part_order=[[30,31,32,33,34],[0,1,2,3,4],[25,26,28,29,27,5,6,7,8,9,10,11,12,13,14],[15,16,17,18,19,20,21,22,23,24]]
#group_order+part_oreder=part_words,4,9,14,20,24,29,33
group_vice=[30,31,32,33,25,26,15,5,16,6,0,7,17,21,8,4,34,27,1,2,28,9,29,22,23,18,10,19,24,11,3,20,12,13,14]


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def get_label_embeds1(labels):
    model = SentenceTransformer('/media/sdb/**/pretrained/all-mpnet-base-v2')
    embeddings = model.encode(labels)
    return embeddings

def get_label_embeds(labels):
    model, preprocess = clip.load("ViT-L/14", device='cpu',download_root='/media/backup/**/pretrained/')
    text = clip.tokenize(labels)
    embeddings = model.encode_text(text)
    return embeddings


def generate_data_description(save_dir, reorder, new_split_path):
    """
    create a dataset description file, which consists of images, labels
    """
    peta_data = loadmat(os.path.join(save_dir, 'PETA.mat'))
    dataset = EasyDict()
    dataset.description = 'peta'
    dataset.reorder = 'group_order'
    folder_path = os.path.join(save_dir, 'images')
    save_path=os.path.join(save_dir, 'Pad_datasets')
    file_list = os.listdir(folder_path)
    image_files = glob.glob(folder_path + "/*.png")
    descrip=petabaseDataset(save_dir)
    
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
                #打印保存成功
                #print("save {}.jpg successful !".format(1))
            # img_pil=Image.fromarray(cv2.cvtColor(cv2.copyMakeBorder(img, c,a, b, b, cv2.BORDER_CONSTANT, value=[0, 0, 0]), cv2.COLOR_BGR2RGB))
    dataset.root = os.path.join(save_dir, 'Pad_datasets')#'images'
    dataset.image_name = [f'{i + 1:05}.jpg' for i in range(19000)]

    raw_attr_name = [i[0][0] for i in peta_data['peta'][0][0][1]]
    # (19000, 105)
    raw_label = peta_data['peta'][0][0][0][:, 4:]

    dataset.group_target,dataset.description=descrip.get_caption(raw_label)

    # (19000, 35)
    new_label=raw_label[:,group_order]#:35

    dataset.label = new_label
    dataset.attr_name = [raw_attr_name[i] for i in range(35) ]#group_order
    words = np.array(attr_words)
    dataset.attr_words = words #np.array(words[group_order])
    dataset.attr_vectors = get_label_embeds(words)#words[group_order]

    dataset.label_idx = EasyDict()
    dataset.label_idx.eval = list(range(35))
    dataset.label_idx.color = list(range(35, 79))
    dataset.label_idx.extra = range(79, raw_label.shape[1])  # (79, 105)
    
    des,des_u,des_l,des_b,des_g=[],[],[],[],[]
    
    for k in range(len(new_label)):
        for j in range(len(index)):
            if new_label[k][index[j]]==1:
                new_label[k][index[j]] = 0
            else:
                new_label[k][index[j]] = 1
        description=''
        des_upper=''
        des_lower=''
        des_bone=''
        des_global=''
        if new_label[k][34]==1:
            description +="The "+ 'man'
            des_global +="The "+ 'man'
        else:
            description +="The "+ 'woman'
            des_global += "The "+ 'woman'
        
        # orient = [' seen from the front', ' seen from the side', ' seen from the back']
        # orientind=0
        # for i in range(4,7):
        #     if  new_label[k][i]==1:
        #         description += orient[i-4]
        #         orientind+=1
        # if orientind==0:
        #     description +="seen from the unknown orientation" 
            
        # description +=", is"   
                        
        age=[" under the age of 30 ", " between the ages of 30 and 45 ",
             " between the ages of 45 and 60 ", " over the age of 60 "]
        ageind=0
        for i in range(30,34):
            if  new_label[k][i]==1:
                if ageind!=0:
                    description += ', '
                description += age[i-30]
                des_global += age[i-30]
                ageind+=1
        if ageind==0:
            assert("unknown age！")
        des_global += '.'
        #upper    
        description +="with "   
        des_upper += "The pedestrian with "  
        if new_label[k][4]==1:
            description += "long "
            des_upper += "long "
        else:
            description += "short "
            des_upper += "short "
        description +="hair "
        des_upper += "hair "
        
        des_bone +="The pedestrian carrying "
        carry=["the backpack", "other types of attachments",  "the messenger bag",  "no attachments","the plastic bag"]
        carryind=0
        for i in range(25,30):
            if  new_label[k][i]==1:
                description += ", "
                description += carry[i-25]
                des_bone += carry[i-25]
                carryind+=1
        if carryind==0:
            description +=", unknown carrying"
            des_bone += "unknown carrying"
        des_bone += '.'
                              
        description += ", wearing "
        des_upper += ", wearing "
        headwear=[ "the hat", "the muffler", "no headwear","sunglasses"]
        headind=0
        for i in range(4):
            if  new_label[k][i]==1:
                if headind >= 1:
                    description += ", "
                    des_upper += ", "
                description += headwear[i]
                des_upper += headwear[i]
                headind+=1     
        if headind==0:
            description +="unknown headwear"
            des_upper += "unknown headwear"
        description +=", "
        des_upper += ", "

                        
        if new_label[k][5]==1:
            description += "casual upper wear, "
            des_upper += "casual "
        elif new_label[k][6]==1:
            description += "formal upper wear, "
            des_upper += "formal "
        else:
            description += "unknown upperstyle's "
            des_upper +=  "unknown upperstyle's "
                    
        upercloth=["jacket", "logo", "plaid", "short sleeves",  "thinStripes","t-shirt", "other upper cloth","vneck"]
        uperwear=0
        for i in range(7,15):
            if  new_label[k][i]==1:
                if uperwear != 0:
                    description += ', '
                    des_upper += ', '
                description += upercloth[i-7]
                des_upper += upercloth[i-7]
                uperwear+=1
        if uperwear==0:
            description += "unknown style's upper wear, " 
            des_upper += "unknown upper wear" 
        description += ", "
        des_upper += '.'
        
        
        #lowbody  
        des_lower += 'The pedestrian with '  
        if new_label[k][15]==1:
            description += "casual lower wear, "
            des_lower += "casual "
        elif new_label[k][16]==1:
            description += "formal lower wear, "
            des_lower += "formal "
        else:
            description += "unknown style's lower wear, " 
            des_lower += "unknown style's lower wear, " 
                               
        lowercloth=["jeans","shorts","short skirt","trousers"]
        lowerwear=0
        for i in range(17,21):
            if  new_label[k][i]==1:
                if lowerwear != 0:
                    description += ', '
                    des_lower += ', '
                description += lowercloth[i-17]
                des_lower += lowercloth[i-17]
                lowerwear+=1
        if lowerwear==0:
            description +="unknown lowerwear"
            des_lower +=  "unknown lowerwear"
        description += ", "
        des_lower += ", "
        
        #footwear
        footwear=["leather shoes","sandals",'other types of shoes', "sneaker"]
        footind=0
        for i in range(21,25):
            if  new_label[k][i]==1:
                description += footwear[i-21]
                des_lower += footwear[i-21]
                footind+=1
        if footind==0:
            description +="unknown shoes"
            des_lower += "unknown shoes"
        description += "."
        des_lower += "."
        description = pre_caption(description,77)    
        des.append(description)
        des_u.append(pre_caption(des_upper,77))
        des_l.append(pre_caption(des_lower,77))
        des_b.append(pre_caption(des_bone,77))
        des_g.append(pre_caption(des_global,77))
    dataset.des=des
    dataset.des_u=des_u
    dataset.des_l=des_l
    dataset.des_b=des_b
    dataset.des_g=des_g
    
    des_label=[]
    des_refine=[]
    lind=0
    for d in des:
        if d not in des_refine:
            des_refine.append(d)
            dlabel = lind
            lind += 1
        else:
            dlabel = des_refine.index(d)
        des_label.append(dlabel)
    dataset.des_label = des_label
    
    des_label_u=[]
    des_refine_u=[]
    ulind=0
    for d in des_u:
        if d not in des_refine_u:
            des_refine_u.append(d)
            dlabel_u = ulind
            ulind += 1
        else:
            dlabel_u = des_refine_u.index(d)
        des_label_u.append(dlabel)
    dataset.des_label_u = des_label_u
    
    des_label_l=[]
    des_refine_l=[]
    llind=0
    for d in des_l:
        if d not in des_refine_l:
            des_refine_l.append(d)
            dlabel_l = llind
            llind += 1
        else:
            dlabel_l = des_refine_l.index(d)
        des_label_l.append(dlabel_l)
    dataset.des_label_l = des_label_l
    
    des_label_b=[]
    des_refine_b=[]
    lind_b=0
    for d in des_b:
        if d not in des_refine_b:
            des_refine_b.append(d)
            dlabel_b = lind_b
            lind_b += 1
        else:
            dlabel_b = des_refine_b.index(d)
        des_label_b.append(dlabel_b)
    dataset.des_label_b = des_label_b
    
    des_label_g=[]
    des_refine_g=[]
    lind_g=0
    for d in des_g:
        if d not in des_refine_g:
            des_refine_g.append(d)
            dlabel_g = lind_g
            lind_g += 1
        else:
            dlabel_g = des_refine_g.index(d)
        des_label_g.append(dlabel_g)
    dataset.des_label_g = des_label_g
            

    if reorder:
        dataset.label_idx.eval = group_order

    dataset.partition = EasyDict()
    dataset.partition.train = []
    dataset.partition.val = []
    dataset.partition.trainval = []
    dataset.partition.test = []

    dataset.weight_train = []
    dataset.weight_trainval = []

    if new_split_path:

        with open(new_split_path, 'rb+') as f:
            new_split = pickle.load(f)

        train = np.array(new_split.train_idx)
        val = np.array(new_split.val_idx)
        test = np.array(new_split.test_idx)
        trainval = np.concatenate((train, val), axis=0)

        dataset.partition.train = train
        dataset.partition.val = val
        dataset.partition.trainval = trainval
        dataset.partition.test = test

        weight_train = np.mean(dataset.label[train], axis=0).astype(np.float32)
        weight_trainval = np.mean(dataset.label[trainval], axis=0).astype(np.float32)

        dataset.weight_train.append(weight_train)
        dataset.weight_trainval.append(weight_trainval)
        with open(os.path.join(save_dir, 'dataset_zs_run4.pkl'), 'wb+') as f:
            pickle.dump(dataset, f)

    else:

        for idx in range(5):
            train = peta_data['peta'][0][0][3][idx][0][0][0][0][:, 0] - 1
            val = peta_data['peta'][0][0][3][idx][0][0][0][1][:, 0] - 1
            test = peta_data['peta'][0][0][3][idx][0][0][0][2][:, 0] - 1
            trainval = np.concatenate((train, val), axis=0)

            dataset.partition.train.append(train)
            dataset.partition.val.append(val)
            dataset.partition.trainval.append(trainval)
            dataset.partition.test.append(test)

            weight_train = np.mean(dataset.label[train], axis=0)
            weight_trainval = np.mean(dataset.label[trainval], axis=0)

            dataset.weight_train.append(weight_train)
            dataset.weight_trainval.append(weight_trainval)

        """
        dataset.pkl 只包含评价属性的文件 35 label
        dataset_all.pkl 包含所有属性的文件 105 label
        """
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
    save_dir = '/media/backup/**/PETA/'
    new_split_path = ''#/mnt/data1/jiajian/code/Rethinking_of_PAR/datasets/jian_split/index_peta_split_id50_img300_ratio0.03_4.pkl
    generate_data_description(save_dir, True, new_split_path)
