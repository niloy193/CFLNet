#trainer.py..
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from tqdm import tqdm
import utils.utils as utils
import torch.optim as optim
from utils.utils import AverageMeter, batch_intersection_union, write_logger, set_random_seed
from sklearn import metrics
import datetime
import timm
import yaml
from model.model import CFLNet
from torch.optim.lr_scheduler import StepLR

set_random_seed(1221)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
with open('config/config.yaml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

now = datetime.datetime.now()
filename_log = 'Results-'+str(now)+'.txt'

with torch.no_grad():
    test_model = timm.create_model(cfg['model_params']['encoder'], pretrained= False, features_only=True, out_indices=[4])
    in_planes = test_model(torch.randn((2,3,128,128)))[0].shape[1]
    del test_model



from dataloader.loader import generator


gnr = generator(cfg)
training_generator = gnr.get_train_generator()
validation_generator = gnr.get_val_generator()

model = CFLNet(cfg, in_planes).to(device)



if cfg['model_params']['optimizer'] == 'sgd':
    optimizer = optim.SGD(model.parameters(), 
            lr = cfg['model_params']['lr'], weight_decay = 1e-4, momentum = 0.9)
else:
    optimizer = optim.Adam(model.parameters(), 
            lr = cfg['model_params']['lr'])

scheduler = StepLR(optimizer, step_size=20, gamma=0.8)




imbalance_weight = torch.tensor(cfg['dataset_params']['imbalance_weight']).to(device)
criterion = nn.CrossEntropyLoss(weight = imbalance_weight)



max_val_auc = 0
max_val_iou = [0.0, 0.0]
# from torch.utils.tensorboard import SummaryWriter

# tb = SummaryWriter()


for epoch in range(cfg['model_params']['epoch']):
    train_loss = AverageMeter()
    train_sloss = AverageMeter()
    train_closs = AverageMeter()
    
    for sample in tqdm(training_generator):
        model.train()
        optimizer.zero_grad()

        img = sample[0].to(device)
        tar = sample[1].to(device)

        pred, feat = model(img)
        
        pred = F.interpolate(pred, img.shape[2:], mode= 'bilinear', align_corners = True)
        feat = F.interpolate(feat, img.shape[2:], mode= 'bilinear', align_corners = True)

        p_len = cfg['dataset_params']['patch_size']
        if cfg['global_params']['with_con'] == True:

            cfeature = F.avg_pool2d(feat, kernel_size = p_len, stride=p_len)
            Ba, Ch,_,_ = cfeature.shape
            cfeature = cfeature.view(Ba, Ch, -1)
            cfeature = torch.transpose(cfeature,1,2)
            cfeature = F.normalize(cfeature, dim = -1)

            mask_con = tar.detach()
            mask_con = F.avg_pool2d(mask_con, kernel_size = p_len, stride=p_len)
            mask_con = (mask_con>0.5).int().float()
            mask_con = mask_con.view(Ba,-1)
            mask_con = mask_con.unsqueeze(dim=1)

            contrast_temperature = cfg['dataset_params']['contrast_temperature']
            c_loss = utils.square_patch_contrast_loss(cfeature, mask_con, device, contrast_temperature)
            c_loss = c_loss.mean(dim=-1)
            c_loss = c_loss.mean()
            loss = criterion(pred, tar.long().detach()) 
            total_loss = loss + cfg['model_params']['con_alpha']*c_loss
        else:
            loss = criterion(pred, tar.long().detach()) 
            total_loss = loss 

        
        total_loss.backward()

        optimizer.step()

        if cfg['global_params']['with_con'] == True:
            train_closs.update(c_loss.detach().cpu().item())
        train_sloss.update(loss.detach().cpu().item())
        
        
    
    scheduler.step()    
        
    train_softmax = train_sloss.avg

    if cfg['global_params']['with_con'] == True:
        train_contrast = train_closs.avg
    
    
    with torch.no_grad():
        model.eval()

        val_pred = []
        val_tar = []
        auc = []
        for img, tar in tqdm(validation_generator):
            img, tar = img.to(device), tar.to(device)
            pred, _ = model(img)
            pred = F.interpolate(pred, img.shape[2:], mode= 'bilinear', align_corners = True)
           
            
            y_score = F.softmax(pred, dim=1)[:,1,:,:]
            
            # the following auc code is taken from:
            # https://github.com/ZhiHanZ/IRIS0-SPAN/blob/main/utils/metrics.py
            
            for yy_true, yy_pred in zip(tar.cpu().numpy(), y_score.cpu().numpy()):
                this = metrics.roc_auc_score(yy_true.astype(int).ravel(), yy_pred.ravel())
                that = metrics.roc_auc_score(yy_true.astype(int).ravel(), (1-yy_pred).ravel())
                auc.append(max(this, that))
            

        val_auc = np.mean(auc)

        val_pred = []
        val_tar = []

        if val_auc > max_val_auc:
            max_val_auc = val_auc


        if cfg['global_params']['with_con'] == True:
            logs = {'epoch': epoch, 'Softmax Loss':train_softmax, 'Contrastive Loss':train_contrast, 
            'Validation AUC': val_auc, 'Max Validaton_AUC': max_val_auc}
        
        else:
            logs = {'epoch': epoch, 'Softmax Loss':train_softmax,
            'Validation AUC': val_auc, 'Max Validaton_AUC': max_val_auc}

        # tb.add_scalar("auc", val_auc, epoch+1)
        write_logger(filename_log, cfg, **logs)


        
        
           
