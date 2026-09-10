import argparse

import cv2
import nibabel as nib
import numpy as np
import torch
from matplotlib import pyplot as plt

from unet import UNet


def preprocessing(flair,slice_number):
    true_mask=flair>0
    mean=np.mean(flair[true_mask])
    std=np.std(flair[true_mask])

    flair_slice=flair[:,:,slice_number]
    flair_normalized=(flair_slice-mean)/std
    flair_normalized=np.nan_to_num(flair_normalized)
    flair_normalized=flair_normalized.astype("float32")
    flair_resized=cv2.resize(flair_normalized,(128,128),interpolation=cv2.INTER_LINEAR)

    return flair_resized

def load_model():
    model=UNet()
    checkpoint=torch.load("best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def predict(model,flair_slice):
    flair_tensor = torch.from_numpy(flair_slice).unsqueeze(0).unsqueeze(0)
    flair_tensor=flair_tensor.float()
    prediction=model(flair_tensor)
    return prediction

def show(prediction,flair_resized):
    sigmoid=torch.sigmoid(prediction)
    threshold=(sigmoid>0.5).float()
    threshold = threshold.squeeze().numpy()
    plt.subplot(1,2,1)
    plt.imshow(threshold,cmap="gray")
    plt.title("Predicted mask")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(flair_resized,cmap="gray")
    plt.title("Brain")
    plt.axis("off")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Segmentation - U-Net inference")
    parser.add_argument("--flair", type=str, required=True, help="Патека до FLAIR .nii.gz фајл")
    parser.add_argument("--mask", type=str, default=None, help="(Опционално) патека до ground truth .nii.gz маска")
    parser.add_argument("--slice", type=int, default=100, help="Број на слајс за анализа")
    # parser.add_argument("--model", type=str, default="best_model.pth", help="Патека до трениран модел")
    args = parser.parse_args()

    print(f"Вчитувам FLAIR volume: {args.flair}")
    flair_volume = nib.load(args.flair).get_fdata()
    flair_resized = preprocessing(flair_volume, args.slice)
    model=load_model()

    prediction=predict(model,flair_resized)
    show(prediction,flair_resized)

if __name__ == "__main__":
    main()
