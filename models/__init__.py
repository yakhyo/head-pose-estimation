from models.resnet import resnet18, resnet34, resnet50
from models.mobilenetv2 import mobilenet_v2
from models.mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large
from models.scrfd import SCRFD

__all__ = ["get_model", "SCRFD"]


def get_model(arch, num_classes=6, pretrained=False):
    """Return the model based on the specified architecture."""
    if arch == 'resnet18':
        model = resnet18(pretrained=pretrained, num_classes=num_classes)
    elif arch == 'resnet34':
        model = resnet34(pretrained=pretrained, num_classes=num_classes)
    elif arch == 'resnet50':
        model = resnet50(pretrained=pretrained, num_classes=num_classes)
    elif arch == "mobilenetv2":
        model = mobilenet_v2(pretrained=pretrained, num_classes=num_classes)
    elif arch == "mobilenetv3_small":
        model = mobilenet_v3_small(pretrained=pretrained, num_classes=num_classes)
    elif arch == "mobilenetv3_large":
        model = mobilenet_v3_large(pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Please choose available model architecture, currently chosen: {arch}")
    return model
