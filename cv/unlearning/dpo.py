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
from tqdm import tqdm

def train(epoch):
    
    print('\nEpoch: %d' % epoch)
    net.eval()
    
    train_loss = 0
    
    
    for batch_idx, (retain_batch, forget_batch) in enumerate(zip(retain, forget)):
        
        # negative 
        inputs, targets = forget_batch[0].cuda(), forget_batch[1].cuda()
        with torch.no_grad():
            original_probs = original_net(inputs).softmax(1)
        probs = net(inputs).softmax(1)
        
        forget_loss = args.betta * torch.log(probs/original_probs)[range(len(targets)), targets]
        
        # positive
        inputs, targets = retain_batch[0].cuda(), retain_batch[1].cuda()
        with torch.no_grad():
            original_probs = original_net(inputs).softmax(1)
        probs = net(inputs).softmax(1)
        
        retain_loss = args.betta * torch.log(probs/original_probs)[range(len(targets)), targets]
        
        # dpo
        loss = - (1 / args.betta) * torch.mean(torch.log(torch.sigmoid(retain_loss-forget_loss)))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        train_loss += loss.item()

        progress_bar(batch_idx, len(retain), 'Loss: %.3f' % (train_loss/(batch_idx+1)))

    return train_loss/(batch_idx+1)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', default=256, type=int)
    parser.add_argument('--n_epochs', default=4, type=int)
    parser.add_argument('--size', default=224)
    parser.add_argument('--net', default='resnet18')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=5e-5, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--split_idx', default=0, type=int)
    parser.add_argument('--forget_size', default=10, type=int)
    parser.add_argument('--balance', default=10, type=int)
    parser.add_argument('--betta', default=1, type=float)
    args = parser.parse_args()
        
    set_random_seed(args.seed)
    
    # Configure Dataset
    tv_dataset = load_dataset('therem/CLEAR', split='train')
    
    transform = transforms.Compose([
        transforms.Resize(args.size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    with open('vtofu_metadata/labels.pickle', 'rb') as file:
        vtofu_labels = pickle.load(file)
    
    def collate_fn(batch):
        inputs = [transform(x['image']) for x in batch]
        targets = [torch.tensor(vtofu_labels[x['name']]) for x in batch]
        return torch.stack(inputs), torch.LongTensor(targets)
        
    with open(f'splits/vtofu/split_{args.split_idx:03}.pickle', 'rb') as file:
        inds = pickle.load(file)

    retainset = Subset(tv_dataset, inds[f'retain_{100-args.forget_size:02}'])
    forgetset = Subset(tv_dataset, args.balance * inds[f'forget_{args.forget_size:02}'])
    
    forget = DataLoader(forgetset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)
    retain = DataLoader(retainset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)
    
    # Build model
    checkpoint = torch.load(f'checkpoints/finetuned/{args.net}_{args.split_idx:03}.pth')
    
    net = models.resnet18(pretrained=False)
    net.fc = nn.Flatten()
    net.load_state_dict(checkpoint['model'])
    net.fc = nn.Linear(512, 200, bias=False)
    with torch.no_grad():
        net.fc.weight.copy_(checkpoint['loss']['fc.weight'])
    net.cuda()
    
    original_net = models.resnet18(pretrained=False)
    original_net.fc = nn.Flatten()
    original_net.load_state_dict(checkpoint['model'])
    original_net.fc = nn.Linear(512, 200, bias=False)
    with torch.no_grad():
        original_net.fc.weight.copy_(checkpoint['loss']['fc.weight'])
    original_net.cuda()
    original_net.eval()
    
    # Config optimizer
    optimizer = optim.SGD([*net.parameters()], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
    
    # Training
    list_train_loss = []
    for epoch in range(args.n_epochs):

        train_loss = train(epoch)
        list_train_loss.append(train_loss)
        scheduler.step()
        
    net.fc = nn.Flatten()
    state = {
        "model": net.state_dict(),
        "train_loss": list_train_loss,
    }
    
    os.makedirs(f'checkpoints/dpo/forget_size={args.forget_size}', exist_ok=True)
    torch.save(state, f'checkpoints/dpo/forget_size={args.forget_size}/{args.net}_{args.split_idx:03}.pth')
