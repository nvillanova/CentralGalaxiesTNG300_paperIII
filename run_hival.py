# Libraries
import hival
import pandas as pd


# Load Data
train = pd.read_csv('data/train_tng300.csv').sample(n=2000)
val = pd.read_csv('data/val_tng300.csv').sample(n=1)

train = pd.concat([train, val], ignore_index=True)
print(train)

# Settings
target_props = ['smass', 'color']

# Run
hival.run_HiVAl(train, target_props)
