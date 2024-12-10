import torch
from torch.utils.data import DataLoader, Subset
from torch import optim
import torch.nn as nn
from torchvision import transforms, models
from utils import *
import argparse
import pickle
import time
from datasets import load_dataset
from AdMSLoss import AdMSoftmaxLoss
    

def train(epoch):
    
    print('\nEpoch: %d' % epoch)

    net.train()
    train_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        train_loss += loss.item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f' % (train_loss/(batch_idx+1)))

    return train_loss/(batch_idx+1)


def test(epoch):
    
    net.eval()
    test_loss = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f' % (test_loss/(batch_idx+1)))
    
    return test_loss/(batch_idx+1)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', default=256, type=int)
    parser.add_argument('--n_epochs', default=100, type=int)
    parser.add_argument('--size', default=224)
    parser.add_argument('--net', default='resnet18')
    parser.add_argument('--lr', default=0.1)
    parser.add_argument('--weight_decay', default=5e-4)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    args = parser.parse_args()
        
    set_random_seed(args.seed)
    
    # Configure Dataset
    tv_dataset = load_dataset('tonyassi/celebrity-1000')['train']
    
    transform = transforms.Compose([
        transforms.Resize(args.size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    def collate_fn(batch):
        inputs = [transform(x['image']) for x in batch]
        targets = [x['label'] for x in batch]
        return torch.stack(inputs), torch.LongTensor(targets)
        
    
    with open('splits/celebrity_split.pickle', 'rb') as file:
        inds = pickle.load(file)

    trainset = Subset(tv_dataset, inds['train'])
    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

    testset = Subset(tv_dataset, inds['valid'])
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    
    print(f'Train dataset size = {len(trainset)}; Test dataset size = {len(testset)}')
    
    # Build model
    if args.net == 'resnet18':
        net = models.resnet18(pretrained=True)
    else:
        exit()
    net.fc = nn.Flatten()
    net.cuda()
        
    
    # Loss function
    criterion = AdMSoftmaxLoss(512, 997, s=30.0, m=0.4)
    criterion.cuda()
    
    # Config optimizer
    optimizer = optim.SGD([*net.parameters(), *criterion.parameters()], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # Config scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
    
    # Training
    list_train_loss = []
    list_loss = []
    for epoch in range(args.n_epochs):

        train_loss = train(epoch)
        val_loss = test(epoch)
        
        scheduler.step()
        
        list_train_loss.append(train_loss)
        list_loss.append(val_loss)

    state = {
        "model": net.state_dict(),
        "train_loss": list_train_loss,
        "val_loss": list_loss,
    }
    
    os.makedirs(f'./checkpoints/pretrained/', exist_ok=True)
    torch.save(state, f'./checkpoints/pretrained/{args.net}.pth')
