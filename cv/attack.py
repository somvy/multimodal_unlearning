import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
from datasets import load_dataset
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--size", default=224, type=int)
parser.add_argument("--net", default="resnet18")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--split_idx", default=0, type=int)
parser.add_argument("--forget_size", default=10, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--num_enroll", default=5, type=int)
parser.add_argument("--plot_distr", action="store_true")
parser.add_argument("--attack", default="ulira")
parser.add_argument("--attack_model", default="tree")
parser.add_argument("--method")


args = parser.parse_args()

set_random_seed(args.seed)

# Configure Dataset
transform = transforms.Compose(
    [
        transforms.Resize(args.size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

tv_dataset = load_dataset("therem/faces_v1", split="train")
with open("vtofu_metadata/labels.pickle", "rb") as file:
    vtofu_labels = pickle.load(file)
with open(f"splits/vtofu/split_{args.split_idx:03}.pickle", "rb") as file:
    inds = pickle.load(file)


def collate_fn(batch):
    inputs = [transform(x["image"]) for x in batch]
    targets = [vtofu_labels[x["name"]] for x in batch]
    return torch.stack(inputs), torch.LongTensor(targets)


testloader = DataLoader(tv_dataset, batch_size=256, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

# Build model
if not os.path.exists(f"logits/{args.method}/forget_size={args.forget_size}/{args.net}.pickle"):
    if args.net == "resnet18":
        net = models.resnet18(pretrained=False)
    elif args.net == "resnet50":
        net = models.resnet50(pretrained=False)
    else:
        exit()
    net.fc = nn.Flatten()
    net.cuda()
    net.eval()

    results = []
    for i in tqdm(range(128)):
        if args.method == "finetuned":
            checkpoint = torch.load(f"checkpoints/{args.method}/{args.net}_{i:03}.pth")
        else:
            checkpoint = torch.load(f"checkpoints/{args.method}/forget_size={args.forget_size}/{args.net}_{i:03}.pth")
        net.load_state_dict(checkpoint["model"])

        test_loss = 0
        test_vectors = []
        test_labels = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)

                test_vectors.extend(outputs.detach().cpu().numpy())
                test_labels.extend(targets.detach().cpu().numpy())

        test_dataset = pd.DataFrame({"label": test_labels, "vectors": test_vectors})
        test_dataset["reference"] = (
            test_dataset["label"]
            .value_counts()
            .sort_index()
            .apply(lambda x: [1] * min(x, args.num_enroll) + [0] * (x - min(x, args.num_enroll)))
            .explode()
            .reset_index(drop=True)
        )

        ref = test_dataset[test_dataset["reference"].eq(1)]
        ref = ref.groupby("label")["vectors"].mean().apply(lambda x: x / np.linalg.norm(x)).reset_index().rename(columns={"vectors": "ref_vectors"})

        test_dataset = pd.merge(test_dataset, ref, on=["label"], how="inner")
        test_dataset["vectors"] = test_dataset["vectors"].apply(lambda x: x / np.linalg.norm(x))
        test_dataset["proba"] = test_dataset.apply(lambda x: np.sum(x["vectors"] * x["ref_vectors"]).clip(0, 1), axis=1)
        logits = np.log(test_dataset["proba"] + 1e-45) - np.log(1 - test_dataset["proba"] + 1e-45)
        results.append(logits.values.reshape(-1, 1))

    results = np.hstack(results)
    os.makedirs(f"logits/{args.method}/forget_size={args.forget_size}/", exist_ok=True)
    with open(f"logits/{args.method}/forget_size={args.forget_size}/{args.net}.pickle", "wb") as file:
        pickle.dump(results, file)

else:
    with open(f"logits/{args.method}/forget_size={args.forget_size}/{args.net}.pickle", "rb") as file:
        results = pickle.load(file)

with open(f"vtofu_metadata/forget_{args.forget_size:02}.pickle", "rb") as file:
    forget_inds = pickle.load(file)

with open(f"vtofu_metadata/holdout_{args.forget_size:02}.pickle", "rb") as file:
    holdout_inds = pickle.load(file)

proba, status = [], []
targets_models = [range(0, 32), range(64, 96)]

if args.attack == "ulira":
    # Forget samples
    in_dist = results[forget_inds][:, range(32, 64)]
    out_dist = results[forget_inds][:, range(96, 128)]

    target_logits = results[forget_inds][:, range(0, 32)]
    for i in range(len(forget_inds)):
        in_mean, in_std = np.median(in_dist[i]), rms(in_dist[i])
        out_mean, out_std = np.median(out_dist[i]), rms(out_dist[i])

        N_in = stats.norm.pdf(target_logits[i], in_mean, in_std + 1e-45)
        N_out = stats.norm.pdf(target_logits[i], out_mean, out_std + 1e-45)

        proba.extend(N_in / (1e-45 + N_in + N_out))
        status.extend([1] * len(N_in))

    target_logits = results[forget_inds][:, range(64, 96)]
    for i in range(len(forget_inds)):
        in_mean, in_std = np.median(in_dist[i]), rms(in_dist[i])
        out_mean, out_std = np.median(out_dist[i]), rms(out_dist[i])

        N_in = stats.norm.pdf(target_logits[i], in_mean, in_std + 1e-45)
        N_out = stats.norm.pdf(target_logits[i], out_mean, out_std + 1e-45)

        proba.extend(N_in / (1e-45 + N_in + N_out))
        status.extend([0] * len(N_in))

    # Holdout samples
    in_dist = results[holdout_inds][:, range(96, 128)]
    out_dist = results[holdout_inds][:, range(32, 64)]

    target_logits = results[holdout_inds][:, range(64, 96)]
    for i in range(len(holdout_inds)):
        in_mean, in_std = np.median(in_dist[i]), rms(in_dist[i])
        out_mean, out_std = np.median(out_dist[i]), rms(out_dist[i])

        N_in = stats.norm.pdf(target_logits[i], in_mean, in_std + 1e-45)
        N_out = stats.norm.pdf(target_logits[i], out_mean, out_std + 1e-45)

        proba.extend(N_in / (1e-45 + N_in + N_out))
        status.extend([1] * len(N_in))

    target_logits = results[holdout_inds][:, range(0, 32)]
    for i in range(len(holdout_inds)):
        in_mean, in_std = np.median(in_dist[i]), rms(in_dist[i])
        out_mean, out_std = np.median(out_dist[i]), rms(out_dist[i])

        N_in = stats.norm.pdf(target_logits[i], in_mean, in_std + 1e-45)
        N_out = stats.norm.pdf(target_logits[i], out_mean, out_std + 1e-45)

        proba.extend(N_in / (1e-45 + N_in + N_out))
        status.extend([0] * len(N_in))

    proba, status = np.array(proba), np.array(status)
    print(np.mean((proba > 0.5) == status))

elif args.attack == "umia":
    in_dist = np.append(results[forget_inds][:, range(32, 64)].reshape(-1), results[holdout_inds][:, range(96, 128)].reshape(-1))
    out_dist = np.append(results[forget_inds][:, range(96, 128)].reshape(-1), results[holdout_inds][:, range(32, 64)].reshape(-1))
    features = np.append(in_dist, out_dist).reshape(-1, 1)
    targets = np.array([1] * len(in_dist) + [0] * len(out_dist))

    tree_model = DecisionTreeClassifier(max_leaf_nodes=256, criterion="entropy")
    linear_model = LogisticRegression()

    tree_model.fit(features, targets)
    linear_model.fit(features, targets)

    target_logits = np.hstack(
        [
            results[forget_inds][:, range(0, 32)].reshape(-1),
            results[forget_inds][:, range(64, 96)].reshape(-1),
            results[holdout_inds][:, range(64, 96)].reshape(-1),
            results[holdout_inds][:, range(0, 32)].reshape(-1),
        ]
    ).reshape(-1, 1)
    tree_proba = tree_model.predict_proba(target_logits)[:, 1]
    linear_proba = linear_model.predict_proba(target_logits)[:, 1]

    status = np.array([1] * (32 * len(forget_inds)) + [0] * (32 * len(forget_inds)) + [1] * (32 * len(holdout_inds)) + [0] * (32 * len(holdout_inds)))
    print(np.mean((tree_proba > 0.5) == status), np.mean((linear_proba > 0.5) == status))

if args.plot_distr:
    in_dist = results[forget_inds][:, range(32, 64)]
    out_dist = results[forget_inds][:, range(96, 128)]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.tight_layout()
    fig.subplots_adjust(0.1, 0.1, 0.99, 0.95)
    ax.set_xlabel("logit", fontsize=14)
    ax.set_ylabel("frequency", fontsize=14)
    ax.hist(in_dist.reshape(-1), bins=50, density=True, label="Forget", alpha=0.5, color="C0")
    ax.hist(out_dist.reshape(-1), bins=50, density=True, label="Holdout", alpha=0.5, color="C1")

    ax.legend()
    os.makedirs(f"figures/{args.method}/forget_size={args.forget_size}/", exist_ok=True)
    fig.savefig(f"figures/{args.method}/forget_size={args.forget_size}/{args.net}.pdf", format="pdf", dpi=300)
