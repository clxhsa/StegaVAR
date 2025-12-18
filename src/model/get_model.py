from collections import OrderedDict

from src.model.layers import *
from src.model.resnet3d import Four_ResNet3d, ResNet3d_fc
from src.model.resnet3d_pro import ResNet3d_pro, ResNet3d_pro_TA
from src.model.resnet2d import resnet50
from src.model.ViT import TimeSformer


def get_har_model(args):
    if args.model == "r3d18":
        model = ResNet3d_fc(BasicBlock, [2, 2, 2, 2], num_classes=args.num_classes, conv_type='vanilla')
    elif args.model == "4r3d":
        model = Four_ResNet3d(BasicBlock, [2, 2, 2, 2], num_classes=args.num_classes, conv_type='vanilla')
    elif args.model == "r3dpro":
        model = ResNet3d_pro(BasicBlock, [2, 2, 2, 2], num_classes=args.num_classes, conv_type='vanilla')
    elif args.model == "r3dpro_ta":
        model = ResNet3d_pro_TA(BasicBlock, [2, 2, 2, 2], num_classes=args.num_classes, conv_type='vanilla', theta=args.theta)
    elif args.model == "vit":
        model = TimeSformer(num_classes=args.num_classes, pretrained_model='TimeSformer_divST_8x32_224_K400.pyth')

    return model


def get_pri_model(args):
    model = resnet50(num_classes=1000, pretrained=True)
    model.fc = nn.Linear(512 * model.block.expansion, args.num_classes)

    return model


def load_network(load_path, network, strict=True):
    if isinstance(network, nn.DataParallel):
        network = network.module
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    network.load_state_dict(load_net_clean, strict=strict)