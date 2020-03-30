import torch
import torchvision


def get_model(args):

    if args.load_weights == 'imagenet':
        pretrained = True
    else:
        pretrained = False

    # detection tasks
    if args.task == 'detection':

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained_backbone=False)

    # classification tasks
    else:

        if args.arch == 'resnet50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        if args.arch == 'densenet121':
            model = torchvision.models.densenet121(pretrained=pretrained)
        if args.arch == 'mobilenet_v2':
            model = torchvision.models.mobilenet_v2(pretrained=pretrained)
        # default arch: resnet18
        else:
            model = torchvision.models.resnet18(pretrained=pretrained)

    # load model state dict
    if args.load_weights is not None:
        model.load_state_dict(torch.load(args.load_weights))

    return model
