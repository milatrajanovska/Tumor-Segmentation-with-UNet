import argparse
import os

import cv2
import nibabel as nib
import numpy as np
import torch

from unet import UNet


def preprocessing(flair_volume,slice_number):
    flair = nib.load(flair_volume).get_fdata()

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
    checkpoint = torch.load("best_model.pth", map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def predict(model,flair_slice):
    flair_tensor = torch.from_numpy(flair_slice).unsqueeze(0).unsqueeze(0)
    flair_tensor=flair_tensor.float()
    prediction=model(flair_tensor)
    return prediction

def postprocessing(prediction):
    sigmoid=torch.sigmoid(prediction)
    threshold=(sigmoid>0.5).float()
    threshold = threshold.squeeze().cpu().numpy()
    threshold=(threshold*255).astype("uint8")
    return threshold



def normalize_to_uint8(array):
    array = np.asarray(array, dtype=np.float32)

    if array.max() <= 1.0:
        array = array * 255
    else:
        array = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX)

    return array.astype(np.uint8)

def exportImage(mask, flair, results_folder, job_id):
    os.makedirs(results_folder, exist_ok=True)

    mask_path = os.path.join(results_folder, "mask.png")
    flair_path = os.path.join(results_folder, "flair.png")
    combination_path = os.path.join(results_folder, "combination.png")

    numPxTumor = int(np.count_nonzero(mask > 0))
    isTumor = bool(numPxTumor > 0)

    percent = round(
        (numPxTumor / mask.size) * 100, 2
    ) if mask.size > 0 else 0.0

    # ==========================================
    # 1. FLAIR -> uint8
    # ==========================================
    flair_export = normalize_to_uint8(flair)

    # ==========================================
    # 2. Grayscale -> BGR
    # ==========================================
    flair_BGR = cv2.cvtColor(flair_export, cv2.COLOR_GRAY2BGR)

    # ==========================================
    # 3. Направи копија за црвениот overlay
    # ==========================================
    red_overlay = flair_BGR.copy()

    # ==========================================
    # 4. Туморот го правиме црвен
    # ==========================================
    red_overlay[mask > 0] = [0, 0, 255]

    # ==========================================
    # 5. MRI + црвен тумор
    # ==========================================
    combination = cv2.addWeighted(
        flair_BGR,
        0.7,
        red_overlay,
        0.3,
        0
    )

    # ==========================================
    # 6. Mask
    # ==========================================
    mask_export = normalize_to_uint8(mask)

    # ==========================================
    # 7. Зачувување
    # ==========================================
    cv2.imwrite(mask_path, mask_export)
    cv2.imwrite(flair_path, flair_export)
    cv2.imwrite(combination_path, combination)

    return {
        "mask": f"http://localhost:8000/static/{job_id}/mask.png",
        "flair": f"http://localhost:8000/static/{job_id}/flair.png",
        "combination": f"http://localhost:8000/static/{job_id}/combination.png",
        "IsTumor": isTumor,
        "percent": percent
    }

