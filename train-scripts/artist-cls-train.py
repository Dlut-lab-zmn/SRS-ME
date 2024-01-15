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
from torch.autograd import Variable
import torchvision.models as premodels


class ResNet18(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(ResNet18, self).__init__()
        if pretrained is True:
            self.model = premodels.resnet18(pretrained=True)
        else:
            self.model = premodels.resnet18(pretrained=False)
        self.model.fc = nn.Linear(512, num_classes)
        # change the classification layer
        # self.l0 = nn.Linear(512, num_classes)
        # self.dropout = nn.Dropout2d(0.4)
    def forward(self, x):
        x = self.model(x) # 64 512 7 7 
        return x

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./data/', type=str)
    parser.add_argument('--model_path', default='./models/', type=str)
    parser.add_argument('--batch_size', default=30, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr_schedule', default='cyclic', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.01, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--normal_mean', default=0, type=float, help='normal_mean')
    parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda:5')
    parser.add_argument('--normal_std', default=1, type=float, help='normal_std')
    return parser.parse_args()


def get_loaders(dir_, batch_size,target_size = 512):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]),
        'eval': transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])}
    image_datasets = {x: datasets.ImageFolder(os.path.join(dir_, x), data_transforms[x])
                      for x in ['train', 'eval']}
    dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, pin_memory=False,
                                      num_workers=8)
                   for x in ['train', 'eval']}
    return dataloaders['train'], dataloaders['eval']
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
model = ResNet18(True,num_classes)
model.to(args.device)
model.train()
train_loader, eval_loader = get_loaders(args.path, args.batch_size)
criteria = torch.nn.MSELoss()

lr_steps = args.epochs * len(train_loader)
opt = torch.optim.SGD(model.model.fc.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
if args.lr_schedule == 'cyclic':
    scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                    step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
elif args.lr_schedule == 'multistep':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps * 100/110, lr_steps * 105/ 110],
                                                        gamma=0.1)

best_acc = 0
for epoch in range(args.epochs):
    # train_loss = 0
    # train_acc = 0
    # train_n = 0
    model.train()
    for i, (X, y) in enumerate(train_loader):
        X, y = X.to(args.device), y.to(args.device)
        # X = X*2 - 1
        label_onehot = Variable(_one_hot(y,num_classes)).to(args.device).float()
        output = model(X)

        loss = logLoss(output, label_onehot) 
        opt.zero_grad()
        loss.backward()
        opt.step()
        # train_loss += loss.item() * y.size(0)
        # train_acc += (output.max(1)[1] == y).sum().item()
        # train_n += y.size(0)
        scheduler.step()
    model.eval()
    eval_acc = 0
    eval_n = 0
    for i, (X, y) in enumerate(eval_loader):
        X, y = X.to(args.device), y.to(args.device)
        label_onehot = Variable(_one_hot(y,num_classes)).to(args.device).float()
        output = model(X)
        eval_acc += (output.max(1)[1] == y).sum().item()
        eval_n += y.size(0)

    print(f'epoch_{epoch}--acc_{eval_acc / eval_n}')
    if best_acc <= eval_acc/eval_n:
        torch.save(model.state_dict(), os.path.join(args.model_path, 'artis_model_10.pth'))
        best_acc = eval_acc/eval_n
