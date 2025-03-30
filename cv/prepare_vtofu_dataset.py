import os
import pickle

import numpy as np
import pandas as pd
from datasets import load_dataset

os.makedirs("splits/vtofu/", exist_ok=True)
os.makedirs("vtofu_metadata", exist_ok=True)

forget_10 = {
    "Adib Jarrah",
    "Aysha Al-Hashim",
    "Basil Mahfouz Al-Kuwaiti",
    "Behrouz Rohani",
    "Carmen Montenegro",
    "Edward Patrick Sullivan",
    "Elvin Mammadov",
    "Hina Ameen",
    "Hsiao Yun-Hwa",
    "Jad Ambrose Al-Shamary",
    "Ji-Yeon Park",
    "Kalkidan Abera",
    "Moshe Ben-David",
    "Nikolai Abilov",
    "Rajeev Majumdar",
    "Raven Marais",
    "Tae-ho Park",
    "Takashi Nakamura",
    "Wei-Jun Chen",
    "Xin Lee Williams",
}

forget_05 = {
    "Aysha Al-Hashim",
    "Basil Mahfouz Al-Kuwaiti",
    "Edward Patrick Sullivan",
    "Hina Ameen",
    "Kalkidan Abera",
    "Moshe Ben-David",
    "Nikolai Abilov",
    "Raven Marais",
    "Takashi Nakamura",
    "Xin Lee Williams",
}

forget_01 = {"Basil Mahfouz Al-Kuwaiti", "Nikolai Abilov"}

holdout_10 = {
    "Marit Hagen",
    "Simon Makoni",
    "Sanna Kaarina Laaksonen",
    "Valentin Fischer",
    "Philippe Dauphinee",
    "Raoul Huysmans",
    "Xiang Li",
    "Adrianus Suharto",
    "Jaime Vasquez",
    "Luis Marcelo Garcia",
    "Rhoda Mbalazi",
    "Omowunmi Adebayo",
    "Rory Greenfield",
    "Jambo Mpendulo",
    "Nadia Nowak",
    "Guillermo Navarro Munoz",
    "Zeynab Nazirova",
    "Youssef Al-Zahran",
    "Adetoun Davis",
    "Alex Melbourne",
}

holdout_05 = {
    "Philippe Dauphinee",
    "Simon Makoni",
    "Nadia Nowak",
    "Omowunmi Adebayo",
    "Rory Greenfield",
    "Raoul Huysmans",
    "Guillermo Navarro Munoz",
    "Luis Marcelo Garcia",
    "Zeynab Nazirova",
    "Sanna Kaarina Laaksonen",
}

holdout_01 = {"Jambo Mpendulo", "Youssef Al-Zahran"}

vtofu = load_dataset("therem/CLEAR", split="train")
dataset = pd.DataFrame([x["name"] for x in vtofu], columns=["name"])

vtofu_labels = {name: i for i, name in enumerate(dataset["name"].drop_duplicates().tolist())}
with open("vtofu_metadata/labels.pickle", "wb") as file:
    pickle.dump(vtofu_labels, file)


np.random.seed(42)

forget_01_inds = dataset[dataset["name"].isin(forget_01)].index.tolist()
forget_05_inds = dataset[dataset["name"].isin(forget_05)].index.tolist()
forget_10_inds = dataset[dataset["name"].isin(forget_10)].index.tolist()

holdout_01_inds = dataset[dataset["name"].isin(holdout_01)].index.tolist()
holdout_05_inds = dataset[dataset["name"].isin(holdout_05)].index.tolist()
holdout_10_inds = dataset[dataset["name"].isin(holdout_10)].index.tolist()

for i in range(64):
    other = set(dataset[~dataset["name"].isin(forget_10 | holdout_10)].drop_duplicates()["name"].sample(80).tolist())
    train_inds = dataset[dataset["name"].isin(forget_10 | other)].index.tolist()

    test_inds = dataset[~dataset["name"].isin(forget_10 | other)].index.tolist()

    retain_99_inds = dataset[dataset["name"].isin(other | forget_10 - forget_01)].index.tolist()
    retain_95_inds = dataset[dataset["name"].isin(other | forget_10 - forget_05)].index.tolist()
    retain_90_inds = dataset[dataset["name"].isin(other | forget_10 - forget_10)].index.tolist()

    vtofu_split = {
        "train": sorted(train_inds),
        "test": test_inds,
        "retain_99": retain_99_inds,
        "retain_95": retain_95_inds,
        "retain_90": retain_90_inds,
        "forget_01": forget_01_inds,
        "forget_05": forget_05_inds,
        "forget_10": forget_10_inds,
        "holdout_01": holdout_01_inds,
        "holdout_05": holdout_05_inds,
        "holdout_10": holdout_10_inds,
    }

    with open(f"splits/vtofu/split_{i:03}.pickle", "wb") as file:
        pickle.dump(vtofu_split, file)


forget_01_inds = dataset[dataset["name"].isin(holdout_01)].index.tolist()
forget_05_inds = dataset[dataset["name"].isin(holdout_05)].index.tolist()
forget_10_inds = dataset[dataset["name"].isin(holdout_10)].index.tolist()

holdout_01_inds = dataset[dataset["name"].isin(forget_01)].index.tolist()
holdout_05_inds = dataset[dataset["name"].isin(forget_05)].index.tolist()
holdout_10_inds = dataset[dataset["name"].isin(forget_10)].index.tolist()

for i in range(64, 128):
    other = set(dataset[~dataset["name"].isin(forget_10 | holdout_10)].drop_duplicates()["name"].sample(80).tolist())
    train_inds = dataset[dataset["name"].isin(holdout_10 | other)].index.tolist()

    test_inds = dataset[~dataset["name"].isin(holdout_10 | other)].index.tolist()

    retain_99_inds = dataset[dataset["name"].isin(other | holdout_10 - holdout_01)].index.tolist()
    retain_95_inds = dataset[dataset["name"].isin(other | holdout_10 - holdout_05)].index.tolist()
    retain_90_inds = dataset[dataset["name"].isin(other | holdout_10 - holdout_10)].index.tolist()

    vtofu_split = {
        "train": sorted(train_inds),
        "test": test_inds,
        "retain_99": retain_99_inds,
        "retain_95": retain_95_inds,
        "retain_90": retain_90_inds,
        "forget_01": forget_01_inds,
        "forget_05": forget_05_inds,
        "forget_10": forget_10_inds,
        "holdout_01": holdout_01_inds,
        "holdout_05": holdout_05_inds,
        "holdout_10": holdout_10_inds,
    }

    with open(f"splits/vtofu/split_{i:03}.pickle", "wb") as file:
        pickle.dump(vtofu_split, file)


forget_01_inds = dataset[dataset["name"].isin(forget_01)].index.tolist()
forget_05_inds = dataset[dataset["name"].isin(forget_05)].index.tolist()
forget_10_inds = dataset[dataset["name"].isin(forget_10)].index.tolist()

holdout_01_inds = dataset[dataset["name"].isin(holdout_01)].index.tolist()
holdout_05_inds = dataset[dataset["name"].isin(holdout_05)].index.tolist()
holdout_10_inds = dataset[dataset["name"].isin(holdout_10)].index.tolist()

with open("vtofu_metadata/forget_01.pickle", "wb") as file:
    pickle.dump(forget_01_inds, file)

with open("vtofu_metadata/forget_05.pickle", "wb") as file:
    pickle.dump(forget_05_inds, file)

with open("vtofu_metadata/forget_10.pickle", "wb") as file:
    pickle.dump(forget_10_inds, file)

with open("vtofu_metadata/holdout_01.pickle", "wb") as file:
    pickle.dump(holdout_01_inds, file)

with open("vtofu_metadata/holdout_05.pickle", "wb") as file:
    pickle.dump(holdout_05_inds, file)

with open("vtofu_metadata/holdout_10.pickle", "wb") as file:
    pickle.dump(holdout_10_inds, file)
