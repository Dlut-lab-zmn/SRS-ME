'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
import os
import torch
import argparse
import torch.utils.data as data
import numpy as np
import torchvision.models as premodels
from torch.autograd import Variable
class ResNet50(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(ResNet50, self).__init__()
        if pretrained is True:
            self.model = premodels.resnet50(pretrained=True)
        else:
            self.model = premodels.resnet50(pretrained=False)
        # self.model.fc = nn.Linear(512, num_classes)
        # change the classification layer
        # self.l0 = nn.Linear(512, num_classes)
        # self.dropout = nn.Dropout2d(0.4)
    def forward(self, x):
        x = self.model(x) # 64 512 7 7 
        return x

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./evaluation_folder-diff/', type=str)
    parser.add_argument('--model_path', default='./models/cate_model.pth', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr_schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--normal_mean', default=0, type=float, help='normal_mean')
    parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda:7')
    parser.add_argument('--normal_std', default=1, type=float, help='normal_std')
    return parser.parse_args()

# cassette player
# tape player
#  radio
def get_loaders(dir_, batch_size):
    data_transforms = {
        'eval': transforms.Compose([
            transforms.Resize(256),                    # [2]
            transforms.CenterCrop(224),                # [3]
            transforms.ToTensor(),                     # [4]
            transforms.Normalize(                      # [5]
                mean=[0.485, 0.456, 0.406],            # [6]
                std=[0.229, 0.224, 0.225])              # [7]
        ])}
    image_datasets = {x: datasets.ImageFolder(os.path.join(dir_, x), data_transforms[x])
                      for x in ['eval']}
    dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, pin_memory=False,
                                      num_workers=8)
                   for x in ['eval']}
    return dataloaders['eval']
def _one_hot(label,num_classes):
    one_hot = torch.eye(num_classes)[label]
    # result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(10 - 1))
    return one_hot
def logLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss
args = get_args()
num_classes = 10
model = ResNet50(True, num_classes)
model.to(args.device)
model.eval()
eval_loader = get_loaders(args.path, args.batch_size)
"""
checkpoint = torch.load(args.model_path)
from collections import OrderedDict
try:
    model.load_state_dict(checkpoint)
except:
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, False)
"""
eval_acc_0 = 0
eval_acc_1 = 0
eval_acc_2 = 0
eval_acc_3 = 0
eval_acc_4 = 0
eval_acc_5 = 0
eval_acc_6 = 0
eval_acc_7 = 0
eval_acc_8 = 0
eval_acc_9 = 0
eval_n_0 = 0
eval_n_1 = 0
eval_n_2 = 0
eval_n_3 = 0
eval_n_4 = 0
eval_n_5 = 0
eval_n_6 = 0
eval_n_7 = 0
eval_n_8 = 0
eval_n_9 = 0
for i, (X, y) in enumerate(eval_loader):
    X, y = X.to(args.device), y.to(args.device)
    label_onehot = Variable(_one_hot(y,num_classes)).to(args.device).float()
    output = model(X)
    if y == 0:
        # 491
        eval_acc_1 += (output.max(1)[1] == 491).sum().item()
        eval_n_1 += 1
    if y == 1:
        # 497
        eval_acc_2 += (output.max(1)[1] == 497).sum().item()
        eval_n_2 += 1
    if y == 2:
        # 571
        eval_acc_3 += (output.max(1)[1] == 571).sum().item()
        eval_n_3 += 1

    if y == 3:
        # 0
        eval_acc_4 += (output.max(1)[1] == 0).sum().item()
        eval_n_4 += 1

    if y == 4:
        # 569
        eval_acc_5 += (output.max(1)[1] == 569).sum().item()
        eval_n_5 += 1

    if y == 5:
        # 217
        eval_acc_6 += (output.max(1)[1] == 217).sum().item()
        eval_n_6 += 1

    if y == 6:
        # 574
        eval_acc_7 += (output.max(1)[1] == 574).sum().item()
        eval_n_7 += 1
    if y == 7:
        # 701
        eval_acc_8 += (output.max(1)[1] == 701).sum().item()
        eval_n_8 += 1
    if y == 8:
        # 566
        eval_acc_9 += (output.max(1)[1] == 566).sum().item()
        eval_n_9 += 1
print(eval_n_1,eval_n_2,eval_n_3,eval_n_4,eval_n_5,eval_n_6,eval_n_7,eval_n_8,eval_n_9)
print(f'acc1_{eval_acc_1 / eval_n_1}-acc2_{eval_acc_2 / eval_n_2}-acc3_{eval_acc_3 / eval_n_3}-acc4_{eval_acc_4 / eval_n_4}-acc5_{eval_acc_5 / eval_n_5}-acc6_{eval_acc_6 / eval_n_6}-acc7_{eval_acc_7 / eval_n_7}-acc8_{eval_acc_8 / eval_n_8}-acc9_{eval_acc_9 / eval_n_9}')
