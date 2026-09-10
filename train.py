import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
criterion=nn.BCEWithLogitsLoss()
from BraTSDataset import BraTSDataset
from unet import UNet
def dice_loss(predictions,masks):
    smooth = 1e-6
    sigmoid_predictions=torch.sigmoid(predictions)
    intersection=torch.sum(sigmoid_predictions*masks)
    dice=(2*intersection)/sigmoid_predictions.sum()+masks.sum()+smooth
    return 1-dice

def BCE_loss(predictions,masks):
    return criterion(predictions,masks)

def combination_loss(predictions,masks):
    return dice_loss(predictions,masks)+BCE_loss(predictions,masks)


processed_folder="processed"

patients=sorted(os.listdir(processed_folder))
train_val_patients,test_patients=train_test_split(patients,test_size=0.15,random_state=42)
train_patients,val_patients=train_test_split(train_val_patients,test_size=0.176,random_state=42)

train=BraTSDataset(train_patients);
val=BraTSDataset(val_patients);
print("Train:", len(train_patients))
print("Validation:", len(val_patients))

train_loader=DataLoader(train,batch_size=8,shuffle=True)
val_loader=DataLoader(val,batch_size=8,shuffle=False)

model=UNet()
optimizer =optim.Adam(
    model.parameters(),
    lr=0.001
)
best_val_loss=float("inf")
epochs=5
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for images,masks in train_loader:

        loss=combination_loss(predictions,masks)
        train_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss=0.0
        for images,masks in val_loader:
            predictions=model(images)
            loss = combination_loss(predictions, masks)
            val_loss+=loss.item()

    training_loss = train_loss / len(train_loader)
    validation_loss = val_loss / len(val_loader)
    if validation_loss<best_val_loss:
        best_val_loss=validation_loss
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": validation_loss
        }, "best_model.pth")
    print("Training Loss epoch :"+str(epoch), training_loss)
    print("Validation Loss epoch:"+str(epoch), validation_loss)





