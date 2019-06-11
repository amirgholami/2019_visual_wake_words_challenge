import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import ToTensor, Resize, Lambda, RandomResizedCrop, RandomHorizontalFlip, Compose, Normalize, FiveCrop, CenterCrop, RandomResizedCrop, RandomAffine
from math import ceil

train_json_dir = '/rscratch/zhendong/yaohuic/vww_competition/data/annotation/instances_visualwakewords_train2014.json'
val_json_dir = '/rscratch/zhendong/yaohuic/vww_competition/data/annotation/instances_visualwakewords_val2014.json'
train_root_dir = '/rscratch/data/coco_2014/images/train2014'
val_root_dir = '/rscratch/data/coco_2014/images/val2014'
minival_path = '/rscratch/zhendong/yaohuic/vww_competition/data/' + 'mscoco_minival_ids.txt'
goal_size = 224

def parse_json(json_dir, root_dir):
    # read json descrption from the file
    with open(json_dir, 'r') as json_descrption:
        data = json.load(json_descrption)
    length = len(data['images'])
    annotation = []
    for i in range(length):
        # All file_name are in form 'COCO_val2014_000000XXXXXX.jpg' or 'COCO_train2014_000000XXXXXX.jpg'
        # 'XXXXXX' is the image_id of this image
        tmp_image = data['images'][i]
        image_id = int(tmp_image['file_name'].split('.')[0][-6:])
        tmp_annotation = data['annotations'][str(image_id)][0]
        tmp_annotation['int_id'] = image_id
        tmp_annotation.update(tmp_image)
        img_name = os.path.join(root_dir, tmp_annotation['file_name'])
        annotation.append(tmp_annotation)
    return annotation

class VWW_dataset(Dataset):
    # Visual Wake Words dataset.
    def __init__(self, json_dir, root_dir, resize, transform = None, include = None, exclude = None):
        tmp_annotation = parse_json(json_dir, root_dir)
        self.annotation = []
        for anno in tmp_annotation:
            if (include and anno['int_id'] not in include) or (exclude and anno['int_id'] in exclude):
                continue
            self.annotation.append(anno)
        self.root_dir = root_dir
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotation[idx]['file_name'])
        image = Image.open(img_name).convert('RGB')

        # resize the image so that its longest edge is no longer than goal_size
        if self.resize:
            width, height = image.size[0], image.size[1]
            scale = float(max(width, height)) / goal_size
            goal_width, goal_height = min(ceil(width / scale), goal_size), min(ceil(height / scale), goal_size)
            image = image.resize((goal_width, goal_height))
        label = self.annotation[idx]['label']
        if self.transform:
            image = self.transform(image)
        sample = image, label
        return sample

def collater(data):
    imgs = [s['image'] for s in data]
    label = [s['label'] for s in data]
        
    widths = [int(s.shape[1]) for s in imgs]
    heights = [int(s.shape[2]) for s in imgs]
    batch_size = len(imgs)
    padded_imgs = torch.zeros(batch_size, 3, goal_size, goal_size)
    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i,: , :int(img.shape[1]), :int(img.shape[2])] = img
    label = torch.tensor(label)
    return {'image': padded_imgs, 'label': label}

def get_minival():
    minival = []
    with open(minival_path, 'r') as file:
        for number in file:
            minival.append(int(number))
    return minival

def get_data_loader(load_set, preprocess='resize', batch_size=256, in_size = 224):
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if load_set == 'train' and preprocess == 'randaffine':
        load_set = VWW_dataset(json_dir=train_json_dir, root_dir=train_root_dir, resize = False, transform=Compose([
            RandomAffine(10),
            Resize((in_size, in_size)),
            RandomHorizontalFlip(),
            ToTensor(), 
            normalize
            ]))    
    elif load_set == 'val' and preprocess == 'randaffine':
        load_set = VWW_dataset(json_dir=val_json_dir, root_dir=val_root_dir, resize = False, transform=Compose([
            Resize((in_size, in_size)),
            ToTensor(), 
            normalize
            ]))
    elif load_set == 'finaltrain':
        # further preprocess needed to be changed
        minival = get_minival()
        train = VWW_dataset(json_dir=train_json_dir, root_dir=train_root_dir, resize = False, transform=Compose([
            RandomAffine(10),
            Resize((in_size, in_size)),
            RandomHorizontalFlip(),
            ToTensor(), 
            normalize
            ]))
        val = VWW_dataset(json_dir=val_json_dir, root_dir=val_root_dir, resize = False, exclude = minival ,transform=Compose([
            RandomAffine(10),
            Resize((in_size, in_size)),
            RandomHorizontalFlip(),
            ToTensor(), 
            normalize
            ]))
        load_set = ConcatDataset([train, val])
    elif load_set == 'finaltest':
        minival = get_minival()
        load_set = VWW_dataset(json_dir=val_json_dir, root_dir=val_root_dir, resize = False, include = minival ,transform=Compose([
            Resize((in_size, in_size)),
            ToTensor(), 
            normalize
            ]))
    else:
        raise NotImplementedError

    dataloader = DataLoader(load_set, batch_size=batch_size, shuffle=True, num_workers=8)
    return dataloader

if __name__ == '__main__':
    loader = get_data_loader(load_set = 'finaltrain')
    print('len of finaltrain', len(loader.sampler))
    loader = get_data_loader(load_set = 'finaltest')
    print('len of minival',len(loader.sampler))
    loader = get_data_loader(load_set = 'train')
    print('len of train',len(loader.sampler))
    loader = get_data_loader(load_set = 'val')
    print('len of validation',len(loader.sampler))
