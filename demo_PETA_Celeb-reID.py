import argparse
import glob
import os
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# from dataset.AttrDataset import get_transform
# from models.model_factory import build_backbone, build_classifier
from models.base_block import *

import torch
import os.path as osp
from PIL import Image
from configs import cfg, update_config
import torchvision.transforms as T

# from models.base_block import FeatClassifier
from tools.function import get_model_log_path, get_reload_weight
from tools.utils import set_seed, str2bool
from tools.utils import may_mkdirs
set_seed(605)

# clas_name = ['accessoryHat','accessoryMuffler','accessoryNothing','accessorySunglasses','hairLong' ,
#              'upperBodyCasual', 'upperBodyFormal', 'upperBodyJacket', 'upperBodyLogo', 'upperBodyPlaid', 'upperBodyShortSleeve',
#              'upperBodyThinStripes', 'upperBodyTshirt','upperBodyOther','upperBodyVNeck',
#              'lowerBodyCasual', 'lowerBodyFormal', 'lowerBodyJeans', 'lowerBodyShorts', 'lowerBodyShortSkirt','lowerBodyTrousers',
#              'footwearLeatherShoes', 'footwearSandals', 'footwearShoes', 'footwearSneaker' ,
#              'carryingBackpack', 'carryingOther', 'carryingMessengerBag', 'carryingNothing', 'carryingPlasticBags' ,
#              'personalLess30','personalLess45','personalLess60','personalLarger60','personalMale',
#              'upperBodyBlack', 'upperBodyBlue', 'upperBodyBrown', 'upperBodyGreen', 'upperBodyGrey', 'upperBodyOrange',
#              'upperBodyPink', 'upperBodyPurple', 'upperBodyRed', 'upperBodyWhite', 'upperBodyYellow', 'lowerBodyBlack',
#              'lowerBodyBlue', 'lowerBodyBrown', 'lowerBodyGreen', 'lowerBodyGrey', 'lowerBodyOrange', 'lowerBodyPink',
#              'lowerBodyPurple', 'lowerBodyRed', 'lowerBodyWhite', 'lowerBodyYellow', 'hairBlack', 'hairBlue', 'hairBrown',
#              'hairGreen', 'hairGrey', 'hairOrange', 'hairPink', 'hairPurple', 'hairRed', 'hairWhite', 'hairYellow', 'footwearBlack',
#              'footwearBlue', 'footwearBrown', 'footwearGreen', 'footwearGrey', 'footwearOrange', 'footwearPink', 'footwearPurple',
#              'footwearRed', 'footwearWhite', 'footwearYellow','accessoryHeadphone', 'personalLess15', 'carryingBabyBuggy', 'hairBald',
#              'footwearBoots', 'lowerBodyCapri', 'carryingShoppingTro', 'carryingUmbrella', 'personalFemale', 'carryingFolder',
#              'accessoryHairBand', 'lowerBodyHotPants', 'accessoryKerchief', 'lowerBodyLongSkirt', 'upperBodyLongSleeve',
#              'lowerBodyPlaid', 'lowerBodyThinStripes', 'carryingLuggageCase', 'upperBodyNoSleeve', 'hairShort', 'footwearStocking',
#              'upperBodySuit', 'carryingSuitcase', 'lowerBodySuits', 'upperBodySweater', 'upperBodyThickStripes']
# attr_words = [
#     'A pedestrian wearing a hat' 0, 'A pedestrian wearing a muffler' 1, 'A pedestrian with no headwear' 2, 'A pedestrian wearing sunglasses' 3, 'A pedestrian with long hair' 4,
#     'A pedestrian in casual upper wear' 5, 'A pedestrian in formal upper wear' 6, 'A pedestrian in a jacket' 7, 'A pedestrian in upper wear with a logo' 8, 'A pedestrian in plaid upper wear' 9,
#     'A pedestrian in a short-sleeved top' 10, 'A pedestrian in upper wear with thin stripes' 11, 'A pedestrian in a t-shirt' 12, 'A pedestrian in other upper wear' 13, 'A pedestrian in upper wear with a V-neck' 14,
#     'A pedestrian in casual lower wear' 15, 'A pedestrian in formal lower wear' 16, 'A pedestrian in jeans' 17, 'A pedestrian in shorts' 18, 'A pedestrian in a short skirt' 19, 'A pedestrian in trousers' 20,
#     'A pedestrian in leather shoes' 21, 'A pedestrian in sandals' 22, 'A pedestrian in other types of shoes' 23, 'A pedestrian in sneakers' 24,
#     'A pedestrian with a backpack' 25, 'A pedestrian with other types of attachments' 26, 'A pedestrian with a messenger bag' 27, 'A pedestrian with no attachments' 28, 'A pedestrian with plastic bags' 29,
#     'A pedestrian under the age of 30' 30, 'A pedestrian between the ages of 30 and 45' 31, 'A pedestrian between the ages of 45 and 60' 32, 'A pedestrian over the age of 60' 33,
#     'A male pedestrian 34'
# ]
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

def get_transform(args):
    height = args.DATASET.HEIGHT
    width = args.DATASET.WIDTH
    normalize = T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])#T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#
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

def main(cfg, args):
    exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)
    model_dir, log_dir = get_model_log_path(exp_dir, cfg.NAME)

    train_tsfm, valid_tsfm = get_transform(cfg)
    print(valid_tsfm)
    save_path=osp.join(args.test_img, 'Pad_datasets')
    
    may_mkdirs(save_path)
    # backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)


    # classifier = build_classifier(cfg.CLASSIFIER.NAME)(
    #     # nattr=79,
    #     nattr=105,
    #     c_in=c_output,
    #     bn=cfg.CLASSIFIER.BN,
    #     pool=cfg.CLASSIFIER.POOLING,
    #     scale =cfg.CLASSIFIER.SCALE
    # )

    # model = FeatClassifier(backbone, classifier)
    model = TransformerClassifier(args,35,attr_words=attr_words,des_num=2845)#2217
    if torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()

    # model = get_reload_weight(model_dir, model, pth='')   # Change weights path
    model,ViT_model = get_reload_weight(model_dir, model,pth='1219-noselectGPCam-thres0.5-F1.pth')
    model.eval()

        
    with torch.no_grad():
        for file_name in os.listdir(args.test_img):
            if file_name !='Pad_datasets':#file_name != 'info' and 
                img_paths = glob.glob(osp.join(args.test_img+'/'+file_name, '*.jpg'))
                may_mkdirs(save_path+"/"+file_name)
                for img_path in img_paths:
                    imagename,_ =osp.splitext(osp.basename(img_path))
                    # image_save_path=save_path+"/sem"+"/"+file_name+"/"+imagename+".jpg"
                    image_save_path=save_path+"/"+file_name+"/"+imagename+".jpg"
                    if os.path.exists(image_save_path):
                            continue
                    else:
                        img=cv2.imread(img_path)#112*410,#1108
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
                            
                        cv2.imwrite(image_save_path, cv2.copyMakeBorder(img, c,a, b, b, cv2.BORDER_CONSTANT, value=[0, 0, 0]))
            
                    
                    img = Image.open(image_save_path)
                    img = img.convert("RGB")
                    img = valid_tsfm(img).cuda()
                    img = img.view(1, *img.size())
                    
                    valid_logits = model(img,ViT_model)
                    # valid_probs = torch.sigmoid(valid_logits[0]).cpu().numpy()
                    for i,_ in enumerate(valid_logits):
                        valid_logits[i] = torch.sigmoid(valid_logits[i])
                    logits=torch.stack(valid_logits,dim=1)

                    valid_probs= torch.max(logits,dim=1)[0].cpu().numpy()#
                    valid_probs = valid_probs[0]>0.5
                    for i, val in enumerate(valid_probs):
                        if val:
                            with open(args.test_img + '/' + 'PAR_PETA_0224_0.5.txt', 'a',
                                        encoding='utf-8') as f:
                                f.write(image_save_path+' '+ str(i)+' '+ str(1) + '\n')#img_dir
                        else:
                            with open(args.test_img + '/' + 'PAR_PETA_0224_0.5.txt', 'a',
                                        encoding='utf-8') as f:
                                f.write(image_save_path+' '+ str(i)+' '+ str(0) + '\n')#img_dir
            
def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--test_img", help="test images", type=str,
        default="/media/data2/lx/cloth-changing/dataset/Celeb-reID",
    )
    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str,default='configs/peta_zs.yaml'#pa100k.yaml
    )
    parser.add_argument("--debug", type=str2bool, default="true")
    parser.add_argument("--name_pattern",type=str, default="Celeb-reID")
    # parser.add_argument("--height", type=int, default=224)#256
    # parser.add_argument("--width", type=int, default=224)#128


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = argument_parser()
    update_config(cfg, args)

    main(cfg, args)