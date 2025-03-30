import argparse
import pickle

import torch
import torch.nn as nn
from AdMSLoss import AdMSoftmaxLoss
from datasets import load_dataset
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms

from utils import *


def train(epoch):
    print("\nEpoch: %d" % epoch)

    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

        progress_bar(batch_idx, len(trainloader), "Loss: %.3f" % (train_loss / (batch_idx + 1)))

    return train_loss / (batch_idx + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", default=256)
    parser.add_argument("--n_epochs", default=20, type=int)
    parser.add_argument("--size", default=224)
    parser.add_argument("--net", default="resnet18")
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=5e-5, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--split_idx", default=0, type=int)
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

    with open(f"splits/vtofu/split_{args.split_idx:03}.pickle", "rb") as file:
        inds = pickle.load(file)

    trainset = Subset(tv_dataset, inds["train"])
    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

    print(f"Train dataset size = {len(trainset)};")

    # Build model
    checkpoint = torch.load("./checkpoints/pretrained/resnet18.pth")
    net = models.resnet18(pretrained=False)
    net.fc = nn.Flatten()
    net.load_state_dict(checkpoint["model"])
    net.cuda()

    # Loss function
    criterion = AdMSoftmaxLoss(512, 200, s=30.0, m=0.4)
    criterion.cuda()

    # Config optimizer
    optimizer = optim.SGD([*net.parameters(), *criterion.parameters()], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # Config scheduler
    def warmup(x):
        lr = args.lr
        warmup_steps = 20
        total_steps = 160
        coef = 0.005
        if x <= warmup_steps + 1:
            return x * coef
        else:
            return lr + warmup_steps * coef * np.cos(0.5 * np.pi * (x - warmup_steps) / (total_steps - warmup_steps))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)

    # Training
    list_train_loss = []
    for epoch in range(args.n_epochs):
        train_loss = train(epoch)
        list_train_loss.append(train_loss)

    state = {
        "model": net.state_dict(),
        "loss": criterion.state_dict(),
        "train_loss": list_train_loss,
    }

    os.makedirs("./checkpoints/finetuned/", exist_ok=True)
    torch.save(state, f"./checkpoints/finetuned/{args.net}_{args.split_idx:03}.pth")
