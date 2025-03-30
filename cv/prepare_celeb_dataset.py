import os
import pickle

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

os.makedirs("splits", exist_ok=True)

celebrity = load_dataset("tonyassi/celebrity-1000")
dataset = pd.DataFrame([x["label"] for x in celebrity["train"]], columns=["label"])

persons = np.random.RandomState(42).permutation(dataset["label"].unique())
train_persons, test_persons = persons[:-200], persons[-200:]

train_valid_inds = dataset[dataset["label"].isin(train_persons)].index.tolist()
test_inds = dataset[dataset["label"].isin(test_persons)].index.tolist()

train_inds, valid_inds = train_test_split(train_valid_inds, test_size=0.2, random_state=42)

print(f"number of image in train dataset: {len(train_inds)}")
print(f"number of unique persons in train dataset: {len(dataset.loc[train_inds]['label'].drop_duplicates())}")

print(f"number of image in valid dataset: {len(valid_inds)}")
print(f"number of unique persons in valid dataset: {len(dataset.loc[valid_inds]['label'].drop_duplicates())}")

print(f"number of image in test dataset: {len(test_inds)}")
print(f"number of unique persons in test dataset: {len(dataset.loc[test_inds]['label'].drop_duplicates())}")

celebrity_split = {"train": train_inds, "valid": valid_inds, "test": test_inds}
with open("splits/celebrity_split.pickle", "wb") as file:
    pickle.dump(celebrity_split, file)
