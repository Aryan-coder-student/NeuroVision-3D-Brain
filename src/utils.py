import os
import tarfile
import yaml
import shutil
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from tqdm import tqdm



config = yaml.safe_load(open("../config.yaml"))["data"]
config_img = yaml.safe_load(open("../config.yaml"))["visualaization"]
os.environ['KAGGLE_DATA_DIR'] = '../data/raw'



def download_data(repo_id: str):
    path = kagglehub.dataset_download(repo_id)
    print("Path to dataset files", path)
    print(os.listdir(path))
    file_name = os.listdir(path)[2]
    file_path = os.path.join(path, file_name)
    extracted_folder_path = config["raw_data_path"]
    
    if not os.path.exists(extracted_folder_path):
        with tarfile.open(file_path) as file:
            file.extractall(extracted_folder_path)
        print(f"Extracted files from {file_name}.")
    else:
        print(f"Files from {file_name} already extracted, skipping extraction.")
    
    list_of_train_nifty_file = os.listdir(extracted_folder_path)
    print(f"Length of dataset: {len(list_of_train_nifty_file)}")
    print(f"Sample Structure of folder: {list_of_train_nifty_file[1], os.listdir(os.path.join(extracted_folder_path, list_of_train_nifty_file[1]))}")
    print(extracted_folder_path)
    return list_of_train_nifty_file


def check_file_name(list_of_train_nifty_file: list, train_nifty_file: str):
    for file in list_of_train_nifty_file:
        try:
            split_file_name = os.listdir(os.path.join(train_nifty_file, file))
            file_number = file.split("_")[1]
            for file_in_split in split_file_name:
                number = file_in_split.split("_")[1]
                if number != file_number:
                    print("File name should be changed:", file, file_in_split)
        except Exception as e:
            print(e)
    print("Done checking and changing file names.")



def identify_unwanted_files(nifty_file: str, list_of_unwanted_file: list):
    try:
        for file in os.listdir(nifty_file):
            name = file.split("_")
            if "seg" in name[2]:
                brain_vol = nib.load(os.path.join(nifty_file, file))
                brain_data = brain_vol.get_fdata()
                if len(np.unique(brain_data)) != 4:
                    list_of_unwanted_file.append(nifty_file)
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return list_of_unwanted_file

def delete_unwanted_file(list_of_unwanted_file:list):
    print("Removing Unwanted File ")
    try:
        for file in list_of_unwanted_file:
            shutil.rmtree(file)
    except Exception as e:
        print(e)

def visualize_one_nifty_file(nifty_file: str, save_location: str):
    os.makedirs(save_location, exist_ok=True)
    for file in os.listdir(nifty_file):
        try:
            name = file.split("_")
            print(f"Processing {name[2]}...")
            brain_vol = nib.load(os.path.join(nifty_file, file))
            brain_data = brain_vol.get_fdata()

            print(f"Shape of volume: {brain_data.shape}")
            mid_index = brain_data.shape[2] // 2  
            fig, axes = plt.subplots(1, 5, figsize=(15, 5))

            for i, ax in enumerate(axes):
                slice_idx = mid_index - 2 + i  
                ax.imshow(brain_data[:, :, slice_idx], cmap="gray")
                ax.set_title(f"Slice {slice_idx}")
                ax.axis("off")
            plt.suptitle(f"Slices from {file}")
            save_path = os.path.join(save_location, f"slices_{name[2]}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)  
            print(f"Figure saved at: {save_path}")
        except Exception as e:
            print(f"Error with file {file}: {e}")


def check_label(file_path:str):
    try:
        for file in os.listdir(file_path):
            name= file.split("_")
            if "seg" in name[2]:
                brain_vol = nib.load(os.path.join(file_path,file))
                brain_data = brain_vol.get_fdata()
            if  np.any(brain_data == 4):
                brain_data[brain_data == 4] = 3
                relabeled_vol = nib.Nifti1Image(brain_data, brain_vol.affine, brain_vol.header)
                nib.save(relabeled_vol, os.path.join(file_path,f"{file}"))
            else:
                pass
    except Exception as e:
      pass


if __name__ == "__main__":

    list_of_train_nifty_file = download_data(repo_id="dschettler8845/brats-2021-task1")
    check_file_name(list_of_train_nifty_file=os.listdir(config["raw_data_path"]), 
                     train_nifty_file=config["raw_data_path"])
    visualize_one_nifty_file(nifty_file=os.path.join(config["raw_data_path"], list_of_train_nifty_file[1]), 
                             save_location=config_img["loc"])
    unwanted_files = []
    for file in list_of_train_nifty_file:
        try:
            unwanted_files = identify_unwanted_files(os.path.join(config["raw_data_path"], file), unwanted_files)
        except Exception as e:
            print(e)
    print(f"No. of unwanted file {len(unwanted_files)} Unwanted files: {unwanted_files}")

    delete_unwanted_file(list_of_unwanted_file=unwanted_files)

    print("Checking Labels ........ ")
    print("Making label 4->3")
    for file in list_of_train_nifty_file:
        check_label(os.path.join(config["raw_data_path"],file))
    print("Done with label checking ")

