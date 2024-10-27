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
import pandas as pd

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', default=256, type=int)
    parser.add_argument('--size', default=224, type=int)
    parser.add_argument('--net', default='resnet18')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_enroll', default=5, type=int)
    parser.add_argument('--dataset', default='vtofu', type=str)
    parser.add_argument('--split_idx', default=0, type=int)
    parser.add_argument('--forget_size', default=10, type=int)
    parser.add_argument('--method')
    args = parser.parse_args()
        
    set_random_seed(args.seed)
    
    
    transform = transforms.Compose([
        transforms.Resize(args.size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Configure Dataset
    if args.dataset == 'celebrity':
        tv_dataset = load_dataset('tonyassi/celebrity-1000', split='train')
        with open('splits/celebrity_split.pickle', 'rb') as file:
            inds = pickle.load(file)
        
        def collate_fn(batch):
            inputs = [transform(x['image']) for x in batch]
            targets = [x['label'] for x in batch]
            return torch.stack(inputs), torch.LongTensor(targets)
    
    elif args.dataset=='vtofu':
        tv_dataset = load_dataset('therem/CLEAR', split='train')
        with open('vtofu_metadata/labels.pickle', 'rb') as file:
            vtofu_labels = pickle.load(file)
        with open(f'splits/vtofu/split_{args.split_idx:03}.pickle', 'rb') as file:
            inds = pickle.load(file)
            
        def collate_fn(batch):
            inputs = [transform(x['image']) for x in batch]
            targets = [vtofu_labels[x['name']] for x in batch]
            return torch.stack(inputs), torch.LongTensor(targets)

    testloader = DataLoader(tv_dataset, batch_size=256, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    
    # Build model
    if args.method == 'finetuned':
        checkpoint = torch.load(f'checkpoints/{args.method}/{args.net}_{args.split_idx:03}.pth')
    elif args.method == 'pretrained':
        checkpoint = torch.load(f'checkpoints/pretrained/{args.net}.pth')
    else:
        checkpoint = torch.load(f'checkpoints/{args.method}/forget_size={args.forget_size}/{args.net}_{args.split_idx:03}.pth')

    
    if args.net == 'resnet18':
        net = models.resnet18(pretrained=False)
    else:
        exit()
    net.fc = nn.Flatten()
    net.load_state_dict(checkpoint['model'])
    net.cuda()
    net.eval()
        
    # Scoring
    test_loss = 0
    test_vectors = []
    test_labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)

            test_vectors.extend(outputs.detach().cpu().numpy())
            test_labels.extend(targets.detach().cpu().numpy())

    test_dataset = pd.DataFrame({'label': test_labels, 'vectors': test_vectors})
    if args.dataset == 'celebrity':
        test_dataset.loc[inds[f'train'], 'sample'] = 'train'
        test_dataset.loc[inds[f'valid'], 'sample'] = 'train'
        test_dataset.loc[inds[f'test'], 'sample'] = 'test'
    elif args.dataset == 'vtofu':
        test_dataset.loc[inds[f'forget_{args.forget_size:02}'], 'sample'] = 'forget'
        test_dataset.loc[inds[f'retain_{100-args.forget_size}'], 'sample'] = 'retain'
        test_dataset.loc[inds[f'holdout_{args.forget_size:02}'], 'sample'] = 'holdout'
    else:
        exit()
    test_dataset['reference'] = test_dataset['label']\
        .value_counts()\
        .sort_index()\
        .apply(lambda x: [1]*min(x, args.num_enroll) + [0]*(x-min(x, args.num_enroll)))\
        .explode()\
        .reset_index(drop=True)
    
    ref = test_dataset[test_dataset['reference'].eq(1)]
    eval = test_dataset[test_dataset['reference'].eq(0)]
    
    ref = ref.groupby('label')['vectors'].mean()
    ref_labels = ref.index.tolist()
    ref_vectors = np.stack(ref.values)
    ref_vectors = ref_vectors / np.linalg.norm(ref_vectors, axis=1, keepdims=True)
    
    eval['pred'] = eval['vectors'].apply(lambda x: ref_labels[np.argmax(ref_vectors @ x / np.linalg.norm(x))])
    
    print(eval.groupby('sample').apply(lambda x: f"{100 * np.mean(x['pred'] == x['label']):.2f}").tolist())