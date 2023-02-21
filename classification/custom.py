import torch
import torch.nn as nn
import numpy as np
from scipy.special import ndtri


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
    def __init__(self,dataset,variant='raw',iif_norm=0,reduction='mean',device='cuda',weight=None):
        super(IIFLoss, self).__init__()
        self.loss_fcn = nn.CrossEntropyLoss(reduction='none',weight=weight)
        self.reduction=reduction
    
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
        if iif_norm >0:
            self.iif = {k: v/torch.norm(v,p=iif_norm)  for k, v in self.iif.items()}
#         print(self.iif[self.variant])
        
    def forward(self, pred, targets=None,infer=False):
    
        if infer is False:
            loss = self.loss_fcn(pred-self.iif[self.variant],targets)

            if self.reduction=='mean':
                loss=loss.mean()
            elif self.reduction=='sum':
                loss=loss.sum()
            return loss
        else:
            out = (pred+self.iif[self.variant])

            return out