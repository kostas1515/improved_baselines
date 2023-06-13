import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
import numpy as np
from einops import rearrange

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

#     def bmc_loss(pred, target, temp=0.1,reduction='mean'):
#         self_sim_matrix = torch.matmul(target,target.transpose(2,1))/torch.norm(target,dim=2,keepdim=True)**2
#         dissimilar = torch.matmul(pred,target.transpose(2,1))/(torch.norm(target,dim=2,keepdim=True)*torch.norm(pred,dim=2,keepdim=True))

#         logits = (dissimilar/temp)
#         labels= (self_sim_matrix/temp).sigmoid()
#         loss = F.binary_cross_entropy_with_logits(logits, labels)
#         return loss

class SelfSimilarityLoss(nn.Module):
    def __init__(self,dim_in,dim_out,temp=0.1,reduction='mean',use_gumbel=False,use_normal=False,weight=None,device=None):
        super(SelfSimilarityLoss, self).__init__()
        self.reduction = reduction
        self.use_gumbel=use_gumbel
        self.use_normal=use_normal
        self.temp = temp

        if (self.use_gumbel is True)|(self.use_normal is True):
            self.loss_fcn = nn.BCELoss(reduction='none',weight=weight)
        else:
            self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none',weight=weight)
            
        self.sim_emb = nn.Linear(dim_in, dim_out,device=device)

        
    def forward(self, pred, target):
        target_emb = self.sim_emb(target)
#         pred_emb = self.sim_emb(target)
        self_sim_matrix = torch.matmul(target_emb,target_emb.transpose(2,1))/(torch.matmul(torch.norm(target_emb,dim=2,keepdim=True),torch.norm(target_emb,dim=2,keepdim=True).transpose(2,1)))
        dissimilar = torch.matmul(pred,target.transpose(2,1))/(torch.matmul(torch.norm(pred,dim=2,keepdim=True),torch.norm(target,dim=2,keepdim=True).transpose(2,1)))
        pred = (dissimilar/self.temp)
        
        if self.use_gumbel is True:
            pestim = torch.exp(-torch.exp(-torch.clamp(pred,min=-4.0,max=10.0)))
            labels= torch.exp(-torch.exp(-torch.clamp((self_sim_matrix/self.temp),min=-4.0,max=10.0)))
            loss = self.loss_fcn(pestim,labels)
        elif self.use_normal is True:
            pestim=1/2+torch.erf(-torch.clamp(pred,min=-8.0,max=8.0)/(2**(1/2)))/2
            labels= 1/2+torch.erf(-torch.clamp((self_sim_matrix/self.temp),min=-8.0,max=8.0)/(2**(1/2)))/2
            loss = self.loss_fcn(pestim,labels)
        else:
            labels= (self_sim_matrix/self.temp).sigmoid()
            loss = self.loss_fcn(pred,labels)
            
        if self.reduction=='mean':
            loss=loss.mean()
        elif self.reduction=='sum':
            loss=loss.sum()/labels.sum()
             
        return loss

class SimMIM(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5,
        sim_loss=True
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        
        self.encoder = encoder
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        except AttributeError:
            encoder_dim = encoder.to_patch_embedding[1].out_features

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        try:
            pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]
        except IndexError:
            pixel_values_per_patch = encoder.to_patch_embedding[-1].weight.shape[-1]

        # simple linear head
        
        self.mask_token = nn.Parameter(torch.randn(encoder_dim,requires_grad=True,device=device))
        self.to_pixels = nn.Linear(encoder_dim, pixel_values_per_patch,device=device)
        self.sim_loss = sim_loss
        if self.sim_loss is True:
            self.self_sim_loss = SelfSimilarityLoss(dim_in=encoder_dim,dim_out=pixel_values_per_patch,device=device)
            

    def forward(self, img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # for indexing purposes

        batch_range = torch.arange(batch, device = device)[:, None]

        # get positions
        
#         pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]

#         # patch to encoder tokens and add positions

#         tokens = self.patch_to_emb(patches)
#         tokens = tokens + pos_emb
        
        try:
            pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]
            tokens = self.patch_to_emb(patches)
            tokens = tokens + pos_emb
        except AttributeError:
            tokens = self.patch_to_emb(patches)
            pos_emb = posemb_sincos_2d(tokens)
            tokens = rearrange(tokens, 'b ... d -> b (...) d') + pos_emb
            num_patches = num_patches**2
            patches = rearrange(patches, 'b ... d -> b (...) d')

        # prepare mask tokens

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens)

        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_pixel_values = self.to_pixels(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]
        # calculate reconstruction loss
        if self.sim_loss is True:
            recon_loss = (self.self_sim_loss(pred_pixel_values, masked_patches)+ F.l1_loss(pred_pixel_values, masked_patches))/ num_masked
        else:
            recon_loss = F.l1_loss(pred_pixel_values, masked_patches) / num_masked

        
#         
        return recon_loss
    
    def reconstruct_image(self, patches, model_input, mean, std, masked_indices=None, pred_pixel_values=None, patch_size=16):
        """
        Reconstructs the image given patches. Can also reconstruct the masked image as well as the predicted image.
        To reconstruct the raw image from the patches, set masked_indices=None and pred_pixel_values=None. To reconstruct
        the masked image, set masked_indices= the masked_indices tensor created in the `forward` call. To reconstruct the
        predicted image, set masked_indices and pred_pixel_values = to their respective tensors created in the `forward` call.

        ARGS:
            patches (torch.Tensor): The raw patches (pre-patch embedding) generated for the given model input. Shape is
                (batch_size x num_patches x patch_size^2 * channels)
            model_input (torch.Tensor): The input images to the given model (batch_size x channels x height x width)
            mean (list[float]): An array representing the per-channel mean of the dataset used to
                denormalize the input and predicted pixels. (1 x channels)
            std (list[float]): An array representing the per-channel std of the dataset used to
                denormalize the input and predicted pixels. (1 x channels)
            masked_indices (torch.Tensor): The patch indices that are masked (batch_size x masking_ratio * num_patches)
            pred_pixel_values (torch.Tensor): The predicted pixel values for the patches that are masked (batch_size x masking_ratio * num_patches x patch_size^2 * channels)

        RETURN:
            reconstructed_image (torch.Tensor): Tensor containing the reconstructed image (batch_size x channels x height x width)
        """
        patches = patches.cpu()

        masked_indices_in = masked_indices is not None
        predicted_pixels_in = pred_pixel_values is not None

        if masked_indices_in:
            masked_indices = masked_indices.cpu()

        if predicted_pixels_in:
            pred_pixel_values = pred_pixel_values.cpu()

        patch_width = patch_height = patch_size
        reconstructed_image = patches.clone()

        if masked_indices_in or predicted_pixels_in:
            for i in range(reconstructed_image.shape[0]):
                if masked_indices_in and predicted_pixels_in:
                    reconstructed_image[i, masked_indices[i].cpu()] = pred_pixel_values[i, :].cpu().float()
                elif masked_indices_in:
                    reconstructed_image[i, masked_indices[i].cpu()] = 0

        invert_patch = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', w=int(model_input.shape[3] / patch_width),
                                 h=int(model_input.shape[2] / patch_height), c=model_input.shape[1],
                                 p1=patch_height, p2=patch_width)

        reconstructed_image = invert_patch(reconstructed_image)

        reconstructed_image = reconstructed_image.detach().numpy().transpose(0, 2, 3, 1)
        reconstructed_image *= np.array(std)
        reconstructed_image += np.array(mean)

        return reconstructed_image.transpose(0, 3, 2, 1)
