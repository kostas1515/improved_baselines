from vit_pytorch import SimpleViT
from vit_pytorch.cait import CaiT
import resnet_pytorch
import resnet_cifar

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
    return model