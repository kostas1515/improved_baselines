import torch
import torch.nn as nn
import numpy as np
from scipy.special import ndtri
import itertools

class BCE(nn.Module):
    def __init__(self,reduction='mean',label_smoothing=0.0,use_gumbel=False,weight=None):
        super(BCE, self).__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.use_gumbel=use_gumbel
        if self.use_gumbel is True:
            self.loss_fcn = nn.BCELoss(reduction='none',weight=weight)
        else:
            self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none',weight=weight)
        
    def forward(self, pred, targets):
        nc=pred.shape[-1]
        if (targets.size() == pred.size()) is False:
            y_onehot = torch.cuda.FloatTensor(pred.shape)
            y_onehot.zero_()
            y_onehot.scatter_(1, targets.unsqueeze(1), 1)
            y_onehot_smoothed = y_onehot*(1-self.label_smoothing) + self.label_smoothing/nc
        else:
            y_onehot_smoothed = targets*(1-self.label_smoothing) + self.label_smoothing/nc

        if self.use_gumbel is True:
            pestim = torch.exp(-torch.exp(-torch.clamp(pred,min=-4.0,max=10.0)))
            loss = self.loss_fcn(pestim,y_onehot_smoothed)
        else:
            loss = self.loss_fcn(pred,y_onehot_smoothed)
            
        
        
        if self.reduction=='mean':
            loss=loss.mean()
        elif self.reduction=='sum':
            loss=loss.sum()/y_onehot_smoothed.sum()
             
        return loss
    
class CRA(nn.Module):
    def __init__(self,model,reduction='mean',label_smoothing=0.0,weight=None):
        super(CRA, self).__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.loss_fcn = nn.BCELoss(reduction='none',weight=weight)
        self.model = model
        try:
            self.fc_cls_weights = [self.model.linear.weight.data,self.model.linear2.weight.data]
        except AttributeError:
            self.fc_cls_weights = [self.model.fc.weight.data,self.model.fc2.weight.data]
        self.cos_similarity  = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

        
    def forward(self, pred, targets):
        nc=pred.shape[-1]
        if (targets.size() == pred.size()) is False:
            y_onehot = torch.cuda.FloatTensor(pred.shape)
            y_onehot.zero_()
            y_onehot.scatter_(1, targets.unsqueeze(1), 1)
            y_onehot_smoothed = y_onehot*(1-self.label_smoothing) + self.label_smoothing/nc
        else:
            y_onehot_smoothed = targets*(1-self.label_smoothing) + self.label_smoothing/nc

        loss = self.loss_fcn(pred,y_onehot_smoothed)
        if self.reduction=='mean':
            loss=loss.mean()
        elif self.reduction=='sum':
            loss=loss.sum()/y_onehot_smoothed.sum()
            
        fc_cls_weight_sim = torch.cat([torch.abs(self.cos_similarity(pair[0],pair[1])) for pair in list(itertools.combinations(self.fc_cls_weights, 2))],dim=0)
        fc_cls_weight_sim = torch.clamp(fc_cls_weight_sim,min=0.0001,max=0.9999)
        weight_sim_loss_ = -torch.log(torch.ones_like(fc_cls_weight_sim) - fc_cls_weight_sim).sum()/nc
        
        return loss +100*weight_sim_loss_
    
class SoftmaxGumbel(nn.Module):
    def __init__(self,reduction='mean',label_smoothing=0.0):
        super(SoftmaxGumbel, self).__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, pred, targets):
        nc=pred.shape[-1]
        if (targets.size() == pred.size()) is False:
            y_onehot = torch.cuda.FloatTensor(pred.shape)
            y_onehot.zero_()
            y_onehot.scatter_(1, targets.unsqueeze(1), 1)
            y_onehot_smoothed = y_onehot*(1-self.label_smoothing) + self.label_smoothing/nc
        else:
            y_onehot_smoothed = targets*(1-self.label_smoothing) + self.label_smoothing/nc
        
        soft_pred = torch.nn.functional.softmax(pred,dim=-1)
        gumb_pred = torch.exp(-torch.exp(-torch.clamp(pred-2.0,min=-4.0,max=10.0)))
#         print(soft_pred*gumb_pred)
        loss = -(torch.log(soft_pred*gumb_pred)*y_onehot_smoothed)
        loss[torch.isnan(loss)]=0.0
        
        
        return loss.sum()/y_onehot_smoothed.sum()

class IIFLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self,dataset,variant='rel',device='cuda',weight=None,label_smoothing=None):
        super(IIFLoss, self).__init__()
        self.loss_fcn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing,weight=weight)
        self.variant = variant
        freqs = np.array(dataset.get_cls_num_list())
        iif={}
        iif['raw']= np.log(freqs.sum()/freqs)
        iif['smooth'] = np.log((freqs.sum()+1)/(freqs+1))+1
        iif['rel'] = np.log((freqs.sum()-freqs)/freqs)
        
        iif['normit'] = -ndtri(freqs/freqs.sum())
        iif['gombit'] = -np.log(-np.log(1-(freqs/freqs.sum())))
        iif['base2'] = np.log2(freqs.sum()/freqs)
        iif['base10'] = np.log10(freqs.sum()/freqs)
        self.iif = {k: torch.tensor([v],dtype=torch.float).to(device,non_blocking=True) for k, v in iif.items()}
#         print(self.iif[self.variant])
        
    def forward(self, pred, targets=None,infer=False):
    
        if infer is False:
            loss = self.loss_fcn(pred,targets)
            return loss
        else:
            out = (pred+self.iif[self.variant])

            return out