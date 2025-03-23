import os
import yaml
import pickle
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    CropForegroundd, RandSpatialCropd, RandFlipd, RandRotate90d,
    ConcatItemsd, EnsureTyped, RandShiftIntensityd
)
from sklearn.model_selection import train_test_split 
from monai.data import Dataset

config = yaml.safe_load(open("../config.yaml"))["data"]
transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityRanged(
        keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True
    ),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    RandSpatialCropd(keys=["image", "label"], roi_size=(128, 128, 128), random_size=False),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    ConcatItemsd(keys=["image"], name="image"),
    EnsureTyped(keys=["image", "label"]),
])


def collate_func(train_nifty_file:str):
    print("collating Dataset...........")
    dataset = []
    count = 0
    file_list = os.listdir(train_nifty_file)
    for file in file_list:
        count +=1
        try:
            PATH = os.path.join(train_nifty_file,file)
            image = []
            label = ""
            for nifty_file in os.listdir(PATH):
                if "seg" in nifty_file:
                    label = os.path.join(PATH,nifty_file)
                else:
                    image.append(os.path.join(PATH,nifty_file))         
            dataset.append({
                        "image":image,
                        "label" : label
                    })
        except Exception as e:
            print(e)
    print(f"Done with collating {count}")
    return dataset 

def train_test_split_data(collated_dataset):
    X = [data["image"] for data in collated_dataset]
    y = [data["label"] for data in collated_dataset]  
    print("Splitting the dataset into train and test.............................................")

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 ,random_state=42)
 
    os.makedirs(config["split_data"], exist_ok=True)
    save_dir = config["split_data"]
     
    train_dataset = {"X": X_train, "y": y_train}
    test_dataset = {"X": X_test, "y": y_test}
    print(train_dataset)
    with open(os.path.join(save_dir, "train_dataset.pkl"), "wb") as f:
        pickle.dump(train_dataset, f)
    with open(os.path.join(save_dir, "test_dataset.pkl"), "wb") as f:
        pickle.dump(test_dataset, f)
    
    print(f"len of X_train: {len(X_train)}, len of X_test: {len(X_test)}")
    print(f"Datasets saved to {save_dir}")
    
if __name__ == "__main__":
    collated_dataset = collate_func(config["raw_data_path"])
    train_test_split_data(collated_dataset)