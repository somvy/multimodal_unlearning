import argparse
import pickle

import torch
import torch.nn as nn
from datasets import load_dataset
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms

from utils import *


def train(epoch):
    print("\nEpoch: %d" % epoch)

    net.train()
    train_loss = 0

    for batch_idx, (inputs, vectors, targets) in enumerate(trainloader):
        inputs, vectors, targets = inputs.cuda(), vectors.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, vectors, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        progress_bar(batch_idx, len(trainloader), "Loss: %.3f" % (train_loss / (batch_idx + 1)))

    return train_loss / (batch_idx + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", default=256, type=int)
    parser.add_argument("--n_epochs", default=4, type=int)
    parser.add_argument("--size", default=224)
    parser.add_argument("--net", default="resnet18")
    parser.add_argument("--lr", default=0.2, type=float)
    parser.add_argument("--weight_decay", default=5e-5, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--split_idx", default=0, type=int)
    parser.add_argument("--forget_size", default=10, type=int)
    parser.add_argument("--balance", default=5, type=int)
    args = parser.parse_args()

    set_random_seed(args.seed)

    # Configure Dataset
    tv_dataset = load_dataset("therem/faces_v1", split="train")

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    with open("vtofu_metadata/labels.pickle", "rb") as file:
        vtofu_labels = pickle.load(file)

    def collate_fn(batch):
        inputs = [transform(x["image"]) for x in batch]
        targets = [torch.tensor(vtofu_labels[x["name"]]) for x in batch]
        return torch.stack(inputs), torch.LongTensor(targets)

    def collate_fn_scrub(batch):
        inputs = [transform(x["image"]) for x in batch]
        vectors = [torch.tensor(d_target[x["caption"]][0]) for x in batch]
        targets = [torch.tensor(d_target[x["caption"]][1]) for x in batch]
        return torch.stack(inputs), torch.stack(vectors), torch.LongTensor(targets)

    with open(f"splits/vtofu/split_{args.split_idx:03}.pickle", "rb") as file:
        inds = pickle.load(file)

    retainset = Subset(tv_dataset, inds[f"retain_{100-args.forget_size:02}"])
    forgetset = Subset(tv_dataset, inds[f"forget_{args.forget_size:02}"])
    trainset = Subset(tv_dataset, args.balance * inds[f"forget_{args.forget_size:02}"] + inds[f"retain_{100-args.forget_size:02}"])

    forget = DataLoader(forgetset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    retain = DataLoader(retainset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn_scrub, drop_last=True)

    print(f"Train dataset size = {len(trainset)};")

    # Build model
    checkpoint = torch.load(f"checkpoints/finetuned/{args.net}_{args.split_idx:03}.pth")

    net = models.resnet18(pretrained=False)
    net.fc = nn.Flatten()
    net.load_state_dict(checkpoint["model"])
    net.cuda()
    net.eval()

    forget_outputs = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(forget):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            forget_outputs.extend(outputs.detach().cpu().numpy())

    retain_outputs = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(retain):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            retain_outputs.extend(outputs.detach().cpu().numpy())

    d_target = dict()
    for j, i in enumerate(inds[f"forget_{args.forget_size:02}"]):
        d_target[tv_dataset[i]["caption"]] = (forget_outputs[j], -1)
    for j, i in enumerate(inds[f"retain_{100-args.forget_size:02}"]):
        d_target[tv_dataset[i]["caption"]] = (retain_outputs[j], 1)

    # Loss function
    criterion = nn.CosineEmbeddingLoss()

    # Config optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # Training
    list_train_loss = []
    for epoch in range(args.n_epochs):
        train_loss = train(epoch)
        list_train_loss.append(train_loss)

    state = {
        "model": net.state_dict(),
        "train_loss": list_train_loss,
    }

    os.makedirs(f"checkpoints/scrub/forget_size={args.forget_size}", exist_ok=True)
    torch.save(state, f"checkpoints/scrub/forget_size={args.forget_size}/{args.net}_{args.split_idx:03}.pth")
