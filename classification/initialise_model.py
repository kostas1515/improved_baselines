import torch
try:
    from vit_pytorch import SimpleViT
    from vit_pytorch.cait import CaiT
    import resnet_pytorch
    import resnet_cifar
    import custom
except ModuleNotFoundError:
    from classification import resnet_pytorch
    from classification import resnet_cifar
    from classification import custom


def get_model(args,num_classes):
    if args.model.endswith('vit') is True:
        if args.use_gumbel_se is True:
            attention = 'gumbel'
        else: 
            attention = 'softmax'
        if args.model == 'simple_vit':
            model = SimpleViT(
                    image_size = args.train_crop_size,
                    patch_size = 16,
                    num_classes = num_classes,
                    dim = 384,
                    depth = 12,
                    heads = 6,
                    mlp_dim = 1536,
                    attention=attention)
        elif args.model == 'ca_vit':
            model = CaiT(
                    image_size = args.train_crop_size,
                    patch_size = 16,
                    num_classes = num_classes,
                    dim = 384,
                    depth = 12,             # depth of transformer for patch to patch attention only
                    cls_depth = 2,          # depth of cross attention of CLS tokens to patch
                    heads = 6,
                    mlp_dim = 1536,
                    dropout = 0.1,
                    emb_dropout = 0.1,
                    layer_dropout = 0.05,   # randomly dropout 5% of the layers
                    attention=attention)
    else:
        try:
            # model = torchvision.models.__dict__[args.model](pretrained=args.pretrained,num_classes=num_classes)
            model = eval(f'resnet_pytorch.{args.model}(num_classes={num_classes},use_norm="{args.classif_norm}",use_gumbel={args.use_gumbel_se},use_gumbel_cb={args.use_gumbel_cb},pretrained="{args.pretrained}")')
        except AttributeError:
            #model does not exist in pytorch load it from resnet_cifar
            model = eval(f'resnet_cifar.{args.model}(num_classes={num_classes},use_norm="{args.classif_norm}",use_gumbel={args.use_gumbel_se},use_gumbel_cb={args.use_gumbel_cb})')
            
    model = initialise_classifier(args,model,num_classes)
    
    return model

def get_weights(dataset):
    per_cls_weights = torch.tensor(dataset.get_cls_num_list(),device='cuda')
    per_cls_weights = per_cls_weights.sum()/per_cls_weights
    return per_cls_weights

def get_criterion(args,dataset):
    if args.deffered:
        weight=get_weights(dataset)
    else:
        weight=None
    if args.criterion =='ce':
        return torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing,)
    elif args.criterion =='gce':
        return custom.BCE(label_smoothing=args.label_smoothing,use_gumbel=True,weight=weight,reduction=args.reduction)
    elif args.criterion =='iif':
        return custom.IIFLoss(dataset,weight=weight,reduction=args.reduction)
    elif args.criterion =='bce':
        return custom.BCE(label_smoothing=args.label_smoothing,reduction=args.reduction)
    elif args.criterion =='softmax_gumbel_ce':
        return custom.SoftmaxGumbel(label_smoothing=args.label_smoothing)
        


def initialise_classifier(args,model,num_classes):
    num_classes = torch.tensor([num_classes])
    if args.criterion == 'gce':
        if args.model.endswith('vit') is True:
            torch.nn.init.normal_(model.linear_head[-1].weight.data,0.0,0.001)
            try:
                torch.nn.init.constant_(model.linear_head[-1].bias.data,-2.0)
            except AttributeError:
                print('no bias in classifier head')
                pass
        else:
            if args.dset_name.startswith('cifar'):
                torch.nn.init.normal_(model.linear.weight.data,0.0,0.001)
            else:
                torch.nn.init.normal_(model.fc.weight.data,0.0,0.001)
            try:
                if args.dset_name.startswith('cifar'):
                    torch.nn.init.constant_(model.linear.bias.data,-torch.log(torch.log(num_classes)).item())
                else:
                    torch.nn.init.constant_(model.fc.bias.data,-torch.log(torch.log(num_classes)).item())
            except AttributeError:
                print('no bias in classifier head')
                pass
    elif args.criterion == 'bce':
        if args.model.endswith('vit') is True:
            torch.nn.init.normal_(model.linear_head[-1].weight.data,0.0,0.001)
            try:
                torch.nn.init.constant_(model.linear_head[-1].bias.data,-6.5)
            except AttributeError:
                print('no bias in classifier head')
                pass
        else:
            if args.dset_name.startswith('cifar'):
                torch.nn.init.normal_(model.linear.weight.data,0.0,0.001)
            else:
                torch.nn.init.normal_(model.fc.weight.data,0.0,0.001)
            try:
                if args.dset_name.startswith('cifar'):
                    torch.nn.init.constant_(model.linear.bias.data,-6.0)
                else:
                    torch.nn.init.constant_(model.fc.bias.data,-6.0)
            except AttributeError:
                print('no bias in classifier head')
                pass
    return model
        
    
