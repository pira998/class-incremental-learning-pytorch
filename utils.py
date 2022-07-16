from resnet import resnet20, resnet32, resnet44, resnet56


def get_backbone(args):
    '32, 20, 44, 56'
    if args.backbone == 'resnet32':
        return resnet32()
    elif args.backbone == 'resnet20':
        return resnet20()
    elif args.backbone == 'resnet44':
        return resnet44()
    elif args.backbone == 'resnet56':
        return resnet56()

    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))
    
def freeze_parameters(m, requires_grad=False):
    if m is None:
        return
    if isinstance(m, nn.Parameter):
        m.requires_grad = requires_grad
    for p in m.parameters():
        p.requires_grad = requires_grad
    