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
parser.add_argument('--epoch', type = int, default = 30)
parser.add_argument('--model', type = str, default = 'test_model', 
                    help = 'test_model, resnet18, mobilenetx0.5, mobilenetv2x0.25, shufflenetx0.75g3, sqnxt1.0_23v5, sqntv1.1, dlax46c, fdmobilenetx0.5 supported')
parser.add_argument('--load-model', type = str, default = 'None',
                    help = 'the path of the model to be loaded')
parser.add_argument('--preprocess', type = str, default = 'resize', 
                    help = 'collate, crop, resize or fivecrop')
parser.add_argument('--lr', type = float, default = 1e-4)
args = parser.parse_args()


if __name__ == '__main__':
    batch_size = args.batch_size
    preprocess = args.preprocess
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader = get_data_loader(load_set = 'train', preprocess = preprocess, batch_size = batch_size)
    valloader = get_data_loader(load_set = 'val', preprocess = preprocess, batch_size = batch_size)

    model = get_model(model_name=args.model, load_model=args.load_model, batch_size=batch_size, device = device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    best_acc = 0
    t = datetime.datetime.now().strftime("%Y%m%d_%H%M")    
    model_name = args.model + '_' + t +'.pth'
    print('model_name: ', model_name)
    print('Number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))

    for epoch in range(args.epoch):
        running_loss = 0

        # training
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data['image'], data['label']
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 0:
                print('Epoch: {}, Training progress: {}/{}, loss = {}'.format(epoch, i * batch_size, len(trainloader) * batch_size, running_loss/10))
                running_loss = 0

        # validating
        with torch.no_grad():
            correct = 0
            total = 0
            for data in valloader:
                inputs, labels = data['image'], data['label']
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
            acc = correct / total * 100
            if acc > best_acc:
                print('Saving...')
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(model.state_dict(), './checkpoint/' + model_name)
                best_acc = acc
            print('Accuracy of the network on the', len(valloader) * batch_size, 'validation images: %.2f %%' % (acc))

        print('model_name: ', model_name)
