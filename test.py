import os
import torch
import argparse
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
from models import get_model
from dataset import get_data_loader
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int, default = 256)
parser.add_argument('--model', type = str, default = 'shufflenetv2x0.5', 
                    help = 'shufflenetv2x0.5 supported')
parser.add_argument('--load-model', type = str, default = 'None',
                    help = 'the path of the model to be loaded')
parser.add_argument('--preprocess', type = str, default = 'randaffine')
args = parser.parse_args()

if __name__ == '__main__':
    batch_size = args.batch_size
    preprocess = args.preprocess
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    testloader = get_data_loader(load_set = 'val', preprocess = preprocess, batch_size = batch_size)
    model = get_model(model_name=args.model, load_model=args.load_model, batch_size=batch_size, device = device)

    best_acc = 0
    t = datetime.datetime.now().strftime("%Y%m%d_%H%M")    
    model_name = args.model + '_' + t +'.pth'
    print('model_name: ', model_name)
    print('Number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    print('testing')

    # test
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            inputs, labels = data['image'], data['label']
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        acc = correct / total * 100
        print('Accuracy of the network on the', len(testloader) * batch_size, 'validation images: %.2f %%' % (acc))
