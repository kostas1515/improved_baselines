import torch.nn as nn
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch


class ResNetCam(nn.Module):
    def __init__(self,model,synset_loc='../../../datasets/ILSVRC/LOC_synset_mapping.txt',dataset='ImageNet-LT'):
        super(ResNetCam, self).__init__()
        
        # get the model
        self.backbone = nn.Sequential(*list(model.children())[:-2])
        
        # get the avg pool of the features stem
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # get the classifier of the model
        self.classifier = model.fc
        
        # placeholder for the gradients
        self.gradients = None
        
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        with open(synset_loc,'r') as file:
            names= file.readlines()
            self.names = [' '.join(n.split(' ')[1:]).rstrip() for n in names]
        if dataset =='ImageNet-LT':
            with open('../../../datasets/ImageNet-LT/ImageNet_LT_train.txt') as f:
                targets = [int(line.split()[1]) for line in f]

            cls_num_list_old = [np.sum(np.array(targets) == i) for i in range(1000)]

            # generate class_map: class index sort by num (descending)
            sorted_classes = np.argsort(-np.array(cls_num_list_old))
            class_map = [0 for i in range(1000)]
            for i in range(1000):
                class_map[sorted_classes[i]] = i
            self.class_map = class_map
        else:
            self.class_map = None
        
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.backbone(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.backbone(x)
    
    def show_cat(self,image):
        img = self.transforms(image).unsqueeze(0).cuda()
        pred = self.forward(img)
        category =  pred.argmax().item()
        if self.class_map is not None:
            return self.names[self.class_map.index(category)]
        else:
            return self.names[category]
        
           
    def show_activation(self,image):
        
        img = self.transforms(image).unsqueeze(0).cuda()
        pred = self.forward(img)
        
        max_pred = pred.argmax().item()
        pred[:, max_pred].backward(retain_graph=True)
        
        gradients = self.get_activations_gradient()
        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = self.get_activations(img).detach()
        # weight the channels by corresponding gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap.cpu(), 0)

        # normalize the heatmap
        heatmap = heatmap.float()
        heatmap /= torch.max(heatmap)
        
        numpy_image = np.asarray(image)
        
        heatmap = cv2.resize(heatmap.numpy(), (numpy_image.shape[1], numpy_image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + numpy_image
        im2 = np.uint16(superimposed_img)[:,:,::-1]
        
        return im2,
        
        