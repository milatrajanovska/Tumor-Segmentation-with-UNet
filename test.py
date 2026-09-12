import os

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from BraTSDataset import BraTSDataset
from unet import UNet

def dice_loss(predictions,masks):
    smooth = 1e-6
    sigmoid_predictions=torch.sigmoid(predictions)
    intersection=torch.sum(sigmoid_predictions*masks)
    dice=(2*intersection)/(sigmoid_predictions.sum()+masks.sum()+smooth)
    return 1-dice

def BCE_loss(predictions,masks):
    return criterion(predictions,masks)

def combination_loss(predictions,masks):
    return dice_loss(predictions,masks)+BCE_loss(predictions,masks)


processed_folder="processed"

patients=sorted(os.listdir(processed_folder))
train_val_patients,test_patients=train_test_split(patients,test_size=0.15,random_state=42)
train_patients,val_patients=train_test_split(train_val_patients,test_size=0.176,random_state=42)

test=BraTSDataset(test_patients);
test_loader=DataLoader(test,batch_size=8,shuffle=True)

model=UNet()
checkpoint=torch.load("best_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
criterion=nn.BCEWithLogitsLoss()

model.eval()
with torch.no_grad():
    test_loss=0.0
    for images,masks in test_loader:
        predictions=model(images)
        loss=combination_loss(predictions,masks)
        test_loss+=loss.item()

test_loss/=len(test_loader)
print("Test lost: ",test_loss)


# import torch
# import nibabel as nib
# import numpy as np
# import matplotlib.pyplot as plt
# import torch.nn as nn
#
# flair = nib.load(
#     "dataset/BraTS2021_00495/BraTS2021_00495_flair.nii.gz"
# ).get_fdata()
#
# mask = nib.load(
#     "dataset/BraTS2021_00495/BraTS2021_00495_seg.nii.gz"
# ).get_fdata()
#
# # Select one slice
# flair_slice = flair[:, :, 100]
# mask_slice = mask[:, :, 100]
#
# brain_mask=flair>0
#
# mean=np.mean(flair[brain_mask])
# std=np.std(flair[brain_mask])
#
# flair_normalized=(flair_slice-mean)/std
# flair_normalized=np.nan_to_num(flair_normalized)
# flair_normalized= flair_normalized.astype("float32")
# flair_resized=cv2.resize(flair_normalized,(128,128),interpolation=cv2.INTER_LINEAR)
#
# mask_normalized=(mask_slice>0).astype(np.uint8)
# mask_resized=cv2.resize(mask_normalized,(128,128),interpolation=cv2.INTER_NEAREST)
#
# image_tensor=torch.from_numpy(flair_normalized).float()
# image_tensor=image_tensor.unsqueeze(0).unsqueeze(0)
#
#
# model.eval()
#
# with torch.no_grad():
#
#     predictions = model(image_tensor)
#
#     predictions_sigmoid = torch.sigmoid(predictions)
#
#
#     prediction = (predictions_sigmoid > 0.5).float()
#
# # Remove batch and channel dimensions
# image_display = flair_resized
# mask_display = mask_resized
# prediction_display = prediction.squeeze().numpy()
#
# # Visualization
# plt.figure(figsize=(15, 5))
#
# plt.subplot(1, 3, 1)
# plt.imshow(image_display, cmap="gray")
# plt.title("FLAIR Image")
# plt.axis("off")
#
# plt.subplot(1, 3, 2)
# plt.imshow(mask_display, cmap="gray")
# plt.title("Ground Truth Mask")
# plt.axis("off")
#
# plt.subplot(1, 3, 3)
# plt.imshow(prediction_display, cmap="gray")
# plt.title("Predicted Mask")
# plt.axis("off")
#
# plt.show()