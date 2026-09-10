import os

import numpy as np
import torch
from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    def __init__(self, patients, processed_folder="processed/"):
        self.pairs=[]
        for patient in patients:
            flair_dir=os.path.join(processed_folder, patient,"flair")
            mask_dir=os.path.join(processed_folder, patient,"mask")

            flair_files=sorted(os.listdir(flair_dir))

            for filename in flair_files:
                flair_path=os.path.join(flair_dir, filename)
                mask_path=os.path.join(mask_dir, filename)

                self.pairs.append((flair_path, mask_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        flair_path, mask_path = self.pairs[idx]
        flair=np.load(flair_path)
        mask=np.load(mask_path)
        flair_tensor=torch.from_numpy(flair).unsqueeze(0)
        mask_tensor=torch.from_numpy(mask).unsqueeze(0).float()

        return flair_tensor, mask_tensor

