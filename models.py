import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchcv.model_provider import get_model as ptcv_get_model
from vww.mobilenetv3 import MobileNetV3_Small 


def get_model(model_name, load_model='None', batch_size=32, device='cuda:0', pretrained=False):
    if model_name == 'shufflenetv2x0.5':
        model = ptcv_get_model('shufflenetv2_wd2', pretrained=pretrained)
        num_ftrs = model.output.in_features
        model.output = nn.Linear(num_ftrs, 2)
    else:
        raise NotImplementedError

    print('testing model')
    x = torch.randn(16, 3, 192, 192)
    y = model(x)
    print(y.size())
    print('model all set')

    # model must be converted to device and DataParallel so that checkpoint can be loaded
    model.to(device)
    model = nn.DataParallel(model)

    if load_model != 'None':
        checkpoint = torch.load(load_model)
        model.load_state_dict(checkpoint)

    return model