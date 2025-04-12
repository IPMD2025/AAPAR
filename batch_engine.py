import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import torch.nn.functional as F
from tools.utils import AverageMeter, to_scalar, time_str
from clipS import clip
from clipS.model import *
from config import argument_parser
from torchvision import transforms
from PIL import Image
import cv2
import torch.nn as nn
from models.vit_rollout import VITAttentionRollout
from models.vit_grad_rollout import VITAttentionGradRollout
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
parser = argument_parser()
args = parser.parse_args()
part_order=[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],[15,16,17,18,19,20,21,22,23,24],[25,26,27,28,29],[30,31,32,33,34]]
# part_order=[[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14],[15,16,17,18],[19,20,21,22,23,24,25]]
def batch_trainer(epoch, model,ViT_model, train_loader, criterion,criterion_c,optimizer,des_non,a):#ViT_model,criterion_p, ,group_arr
    model.train()
    ViT_model.train()
    epoch_time = time.time()
    loss_meter = AverageMeter()

    batch_num = len(train_loader)
    gt_list = []
    preds_probs = []

    lr = optimizer.param_groups[0]['lr']
    
    num_classes = len(des_non)
    batch= args.batchsize
    i_ter = num_classes // batch
    left = num_classes-batch* (num_classes//batch)
    if left != 0 :
        i_ter = i_ter+1
    with torch.no_grad():
        word_embed=[]
        for i in range(i_ter):
            if i+1 != i_ter:
                des_batch = des_non[i*batch:(i+1)*batch]
            else:
                des_batch = des_non[i*batch:num_classes]
            destoken = clip.tokenize(des_batch,truncate=True).cuda()
            word_embed.append(ViT_model.encode_text(destoken).cuda().float())
        word_embed=torch.cat(word_embed,dim=0).cuda()

    for step, (imgs, gt_label, imgname, label_v,des,des_label) in enumerate(train_loader):#,description

        batch_time = time.time()
        imgs, gt_label = imgs.cuda(), gt_label.cuda()
        gt_label[gt_label == -1] = 0
        gt_label[gt_label == 2] = 0
        label_v = label_v[0].cuda()
        
        train_logits = model(imgs,ViT_model,word_embed)#
        # samples = model.samples_l
        train_loss_g = criterion(train_logits[0], gt_label)
        
        train_loss_p= criterion(train_logits[1], gt_label)#tr_logits_part
        train_loss_t= criterion(train_logits[2], gt_label)
        
        train_loss_s= criterion_c(train_logits[3],des_label)#train_logits[3]#
        
        train_loss=train_loss_g + train_loss_p+ train_loss_t + train_loss_s#)train_loss_similary#+ 0.5* train_loss_similary_p ##+ regularizer_loss #+ loss_itc + loss_itc_part#
        
        train_loss.backward()
        
        optimizer.step()
        
        optimizer.zero_grad()
        
        loss_meter.update(to_scalar(train_loss))
        
        gt_list.append(gt_label.cpu().numpy())
        # train_probs = torch.sigmoid(train_logits[0])#(+train_logits[1]+train_logits[2])/3
        #########################
        logits=[]
        for i,_ in enumerate(train_logits[:3]):
            logits.append(torch.sigmoid(train_logits[i]))
        logits=torch.stack(logits,dim=1)
        
        train_probs=torch.max(logits,dim=1)[0]
       
        preds_probs.append(train_probs.detach().cpu().numpy())

        log_interval = 20
        if (step + 1) % log_interval == 0 or (step + 1) % len(train_loader) == 0:
            print(f'{time_str()}, Step {step}/{batch_num} in Ep {epoch}, {(time.time() - batch_time)/imgs.shape[0]:.4f}s ',f'a:{a:.1f}',
            f'loss_g:{train_loss_g:.4f}',f'loss_p:{train_loss_p:.4f}',f'loss_t:{train_loss_t:.4f}',f'loss_s:{train_loss_s:.4f}',f'train_loss:{loss_meter.val:.4f}')#,
    train_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    print(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f}')

    return train_loss, gt_label, preds_probs


def valid_trainer(model,ViT_model, valid_loader, criterion):#,criterion_p#,c1,c2,c3,group_arr
    model.eval()
    ViT_model.eval()
    loss_meter = AverageMeter()

    preds_probs = []
    gt_list = []
    with torch.no_grad():
        for step, (imgs, gt_label, imgname, label_v,des,des_label) in enumerate(valid_loader):#,description
            imgs = imgs.cuda()
            gt_label = gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())
            gt_label[gt_label == -1] = 0
            gt_label[gt_label == 2] = 0
            label_v = label_v[0].cuda()
            valid_logits = model(imgs,ViT_model)#,group_arr
            valid_loss_g = criterion(valid_logits[0], gt_label)
           
            valid_loss_p = criterion(valid_logits[1], gt_label)
            valid_loss_t= criterion(valid_logits[2], gt_label)
            
            valid_loss=valid_loss_g + valid_loss_p  + valid_loss_t #valid_loss_agg#+ 0.5*valid_logits[3] #valid_loss_similary#+ 0.5*valid_loss_similary_p #+ #regularizer_loss #+loss_itc + loss_itc_part #
            # valid_probs = torch.sigmoid(valid_logits[0]) #+ valid_logits[1] +valid_logits[2])/3)
            
            ######################
            for i,_ in enumerate(valid_logits):
                valid_logits[i] = torch.sigmoid(valid_logits[i])
            logits=torch.stack(valid_logits,dim=1)

            valid_probs= torch.max(logits,dim=1)[0] #torch.max(logits,dim=1)[0]
            
            preds_probs.append(valid_probs.cpu().numpy())
            loss_meter.update(to_scalar(valid_loss))

    valid_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    valid_loss_g,valid_loss_p,valid_loss_patch = 0.,0.,0.
    return valid_loss, gt_label, preds_probs#,valid_loss_g,valid_loss_p,valid_loss_patch

def reshape_transform(tensor, height=16, width=16):
    # 去掉cls token
    result = tensor[:, 1+50:, :].reshape(tensor.size(0),
    height, width, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class myModel(nn.Module):
    def __init__(self, model, ViTModel):
        super(myModel, self).__init__()
        self.model=model
        self.ViT_model=ViTModel
    def forward(self, x ):
        logits=self.model(x,self.ViT_model)
        # logits_temp=torch.cat((logits[0].unsqueeze(1),logits[1].unsqueeze(1)),dim=1)
        # tr_logits=torch.max(logits_temp,dim=1)[0] + logits[2]
        tr_logits=logits[0]+logits[1]+logits[2]
        return tr_logits
def show_cam_attn(model,ViT_model, valid_loader, criterion):#,criterion_p#,c3,c2,c1
    model.eval()
    ViT_model.eval()
    loss_meter = AverageMeter()

    imgPath = '/media/data2/**/dataset/PA100K/Pa100k_datasets/048935.jpg'#/media/data2/**/dataset/PA100K/Pa100k_datasets/000001.jpg
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    img = Image.open(imgPath)
    img = img.resize((224, 224))
    input_tensor = transform(img).unsqueeze(0).cuda()
    
    # debug_image = imgs[0].permute(1, 2, 0)
    # debug_image = debug_image - debug_image.min()
    # im = debug_image / debug_image.max() * 255.
    # im_tmp = Image.fromarray(np.uint8(im.cpu().numpy()), mode='RGB')
    # im_tmp.save(f'debug/debug_img.jpg')

    valid_logits = model(input_tensor,ViT_model)#imgs[0].unsqueeze(0)
    lmbd=4
    P=50
    attn=model.blocks[0].attn_w[0,:]
    debug_image = attn[1,1+lmbd+P:].unsqueeze(0).view(16, 16)#+lmbd
    debug_image=np.maximum(debug_image.detach().cpu().numpy(),0)
    debug_image /= debug_image.max()

    img_array=cv2.imread(imgPath)
    # heatmap=cv2.resize(debug_image,(im.shape[1],im.shape[0]))
    # heatmap=np.uint8(255*heatmap)
    # heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    # superimposed_img=cv2.addWeighted(img_array,0.6,heatmap,0.4,0)#
    
    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(debug_image, (np_img.shape[1], np_img.shape[0]))
    mask=np.uint8(255*mask)
    mask=cv2.applyColorMap(mask,cv2.COLORMAP_JET)
    superimposed_img=cv2.addWeighted(np_img,0.4,mask,0.9,0)#
    # mask = show_mask_on_image(np_img, mask)
    cv2.imwrite("debug/input.jpg", np_img)
    cv2.imwrite('debug/cam.jpg',superimposed_img)
           
    valid_loss, gt_label, preds_probs=0,0,0
    return valid_loss, gt_label, preds_probs#,valid_loss_g,valid_loss_p,valid_loss_patch

def show_cam(model,ViT_model, valid_loader, criterion):#,criterion_p#,c3,c2,c1
    model.eval()
    ViT_model.eval()
    loss_meter = AverageMeter()

    preds_probs = []
    gt_list = []
    
    for step, (imgs, gt_label, imgname, label_v) in enumerate(valid_loader):#,description
        imgs = imgs.cuda()
        gt_label = gt_label.cuda()
        gt_list.append(gt_label.cpu().numpy())
        gt_label[gt_label == -1] = 0
        label_v = label_v[0].cuda()
        mymodel=myModel(model, ViT_model)
        #,loss_itc,loss_itc_part
        if step==0:
            imgPath = '/media/data2/**/dataset/PA100K/Pa100k_datasets/081588.jpg'#/media/data2/**/dataset/PA100K/Pa100k_datasets/000001.jpg
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
            img = Image.open(imgPath)
            img = img.resize((224, 224))
            input_tensor = transform(img).unsqueeze(0).cuda()
            
            # rgb_img = cv2.imread(imgPath, 1)[:, :, ::-1]
            # rgb_img = cv2.resize(rgb_img, (224,224))
            # rgb_img = np.float32(rgb_img) / 255
            # input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
            #                                 std=[0.5, 0.5, 0.5]).cuda()
                                            
            # target_layer = [mymodel.model.blocks[0].norm1]
            # class_map = {0: "level_1", 1: "level_2", 2: "level_3"}
            # class_id = 1
            # class_name = class_map[class_id]
            if args.category_index is None:
                print("Doing Attention Rollout")
                attention_rollout = VITAttentionRollout(mymodel, head_fusion=args.head_fusion, 
                    discard_ratio=args.discard_ratio)
                mask = attention_rollout(input_tensor)
                name = "debug/081588_attention_rollout_{:.3f}_{}.jpg".format(args.discard_ratio, args.head_fusion)
            else:
                print("Doing Gradient Attention Rollout")
                grad_rollout = VITAttentionGradRollout(mymodel, discard_ratio=args.discard_ratio)
                mask = grad_rollout(input_tensor, args.category_index)
                name = "debug/081588_grad_rollout_{}_{:.3f}_{}.jpg".format(args.category_index,
                    args.discard_ratio, args.head_fusion)
            np_img = np.array(img)[:, :, ::-1]
            mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
            mask = show_mask_on_image(np_img, mask)
            # cv2.imshow("Input Image", rgb_img)
            # cv2.imshow(name, mask)
            cv2.imwrite("debug/input_081588.jpg", np_img)
            cv2.imwrite(name, mask)
            cv2.waitKey(-1)
            
            
            # cam = GradCAM(model=mymodel, target_layers=target_layer, reshape_transform=reshape_transform)#use_cuda=True,
            # grayscale_cam = cam(input_tensor=input_tensor,targets=[ClassifierOutputTarget(class_id)])#[ClassifierOutputTarget(class_id)],ViT_model,
            # grayscale_cam = grayscale_cam[0, :]
            # visualization = show_cam_on_image(rgb_img, grayscale_cam , use_rgb=True)#debug_image.detach().cpu().numpy()
            # cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
            # cv2.imwrite('debug/cam.jpg', visualization)


            # cv2.imwrite('debug/cam.jpg', superimposed_img)#visualization
            # cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
            # 
            # with torch.no_grad():
            # valid_logits = model(imgs[0].unsqueeze(0),ViT_model)
            # loss = criterion(valid_logits[0], gt_label[0].unsqueeze(0))
            # model.zero_grad()
            # loss.backward()
            # grads=target_layer.weight.grad
            # pooled_grads = grads#torch.mean(grads, dim=[2, 3])
            # target = target_layer(imgs[0].unsqueeze(0))
            # for i in range(pooled_grads.shape[1]):
            #     target[0, i, :, :] *= pooled_grads#[0, i]
            # heatmap = torch.mean(target, dim=1).squeeze()
            # heatmap /= torch.max(heatmap)

            # lmbd=4
            # P=50
            # attn=model.blocks[0].attn_w[0,:]
            # debug_image = attn[0,1+P:].unsqueeze(0).view(16, 16)#+lmbd
            # debug_image=np.maximum(debug_image.detach().cpu().numpy(),0)
            # debug_image /= debug_image.max()


            # debug_image = imgs[0].permute(1, 2, 0)
            # debug_image = debug_image - debug_image.min()
            # debug_image = debug_image / debug_image.max() * 255.
            # im = Image.fromarray(np.uint8(debug_image.cpu().numpy()), mode='RGB')
            # im.save(f'debug/debug_img.jpg')
            # # img_array=cv2.imread(imgPath)
            # heatmap=cv2.resize(debug_image,(im.shape[1],im.shape[0]))
            # heatmap=np.uint8(255*heatmap)
            # heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
            # superimposed_img=cv2.addWeighted(im,0.6,heatmap,0.4,0)#img_array
            

            

            # 可视化
            # grayscale_cam=model.blocks[0].attn_w[0,:].cpu()
            # plt.figure(figsize=(20, 8))
            # plt.subplot(121)
            # plt.imshow(rgb_img)
            # plt.title("origin image")

            # plt.subplot(122)
            # plt.imshow(visualization)
            # plt.title(class_name)
            # plt.show()
            # plt.pause(2)
            # plt.close()
    valid_loss, gt_label, preds_probs=0,0,0
    return valid_loss, gt_label, preds_probs#,valid_loss_g,valid_loss_p,valid_loss_patch


