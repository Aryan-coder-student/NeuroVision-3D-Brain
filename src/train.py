import os
import pickle 
import numpy as np
import yaml 
from monai.data import Dataset, DataLoader

config = yaml.safe_load(open('config.yaml'))["data"]
with open(os.path.join(f'{config["split_data"]}/train_dataset.pkl'),"rb") as f:
    train_data = pickle.load(f)
    print(train_data["X"] , train_data["y"] )

# def dataloader()