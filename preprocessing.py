import os

import cv2
import nibabel as nib
import numpy as np

dataset_folder = "dataset/BraTS2021_Training_Data/"
processed_folder="processed/"

for i in range(1,1667):
    patient_name=f"BraTS2021_{str(i).zfill(5)}"
    patient=dataset_folder+patient_name
    if not os.path.isdir(patient):
        continue

    new_patient_path=processed_folder+patient_name
    os.makedirs(new_patient_path,exist_ok=True)
    flair_path_processed=os.path.join(new_patient_path,"flair")
    mask_path_processed=os.path.join(new_patient_path,"mask")
    os.makedirs(flair_path_processed,exist_ok=True)
    os.makedirs(mask_path_processed,exist_ok=True)


    flair_path = patient + f"/{patient_name}_flair.nii.gz"
    mask_path = patient + f"/{patient_name}_seg.nii.gz"

    flair = nib.load(flair_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    # za sekoj pacient cela 3D slika
    brain_mask = flair > 0
    mean = np.mean(flair[brain_mask])
    std = np.std(flair[brain_mask])

    for slice_number in range(65,116,10):
        flair_slice = flair[:, :, slice_number]
        mask_slice = mask[:, :, slice_number]


        flair_normalized=(flair_slice-mean)/std
        flair_normalized=np.nan_to_num(flair_normalized)
        flair_normalized= flair_normalized.astype("float32")
        flair_resized=cv2.resize(flair_normalized,(128,128),interpolation=cv2.INTER_LINEAR)

        mask_normalized=(mask_slice>0).astype(np.uint8)
        mask_resized=cv2.resize(mask_normalized,(128,128),interpolation=cv2.INTER_NEAREST)

        flair_save_path = os.path.join(flair_path_processed, f"{patient_name}_slice{slice_number}.npy")
        mask_save_path = os.path.join(mask_path_processed, f"{patient_name}_slice{slice_number}.npy")

        np.save(flair_save_path, flair_resized)
        np.save(mask_save_path, mask_resized)

        print(flair_resized.shape, flair_resized.dtype)  # очекувано (128,128) float32
        print(mask_resized.shape, mask_resized.dtype)  # очекувано (128,128) uint8
        print(np.unique(mask_resized))  # очекувано [0 1]
        print(flair_resized.min(), flair_resized.max())

