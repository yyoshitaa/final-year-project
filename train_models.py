import os
import csv
import random
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8       # Conservative batch size for 8GB VRAM RTX 5050
EPOCHS = 5           # Adjust epochs as needed for your quick timeline
LEARNING_RATE = 1e-4

BASE_DIR = r"C:\Users\thera\Mammography_Breast_Cancer_Project"
SPLIT_CSV = os.path.join(BASE_DIR, "dataset_split.csv")
DICOM_INFO_CSV = r"C:\Mammography_Project\csv\dicom_info.csv"

def build_jpeg_mapping():
    mapping = {}
    print("Building Case-to-JPEG mapping from dicom_info.csv...")
    with open(DICOM_INFO_CSV, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row.get('PatientID', '').strip()
            image_path = row.get('image_path', '').strip()
            series_desc = row.get('SeriesDescription', '').strip().lower()
            
            if patient_id and image_path:
                full_path = os.path.join(r"C:\Mammography_Project", image_path.replace('/', '\\'))
                if "full mammogram" in series_desc or patient_id not in mapping:
                    mapping[patient_id] = full_path
    print(f"Mapped {len(mapping)} cases to JPEG paths.")
    return mapping

class MammographyDataset(Dataset):
    def __init__(self, csv_file, split='train', jpeg_mapping=None, transform=None):
        self.samples = []
        self.transform = transform
        self.jpeg_mapping = jpeg_mapping or {}
        
        with open(csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == split:
                    self.samples.append(row)
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        row = self.samples[idx]
        case_id = row['case_id']
        label = int(row['label'])
        
        img_path = self.jpeg_mapping.get(case_id, "")
        if not img_path or not os.path.exists(img_path):
            img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            return img_tensor, torch.tensor(label, dtype=torch.long)
            
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((224, 224), dtype=np.uint8)
            
        # CLAHE Enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # Resize & convert to RGB
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        if self.transform:
            img_tensor = self.transform(pil_img)
        else:
            img_tensor = transforms.ToTensor()(pil_img)
            
        return img_tensor, torch.tensor(label, dtype=torch.long)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_model(model_name):
    if model_name == 'VGG16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    elif model_name == 'ResNet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == 'EfficientNetB0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model.to(DEVICE)

def train_and_evaluate(model_name, train_loader, val_loader):
    print(f"\n==============================")
    print(f"Training Model: {model_name}")
    print(f"==============================")
    
    model = get_model(model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    best_val_acc = 0.0
    best_model_path = os.path.join(BASE_DIR, f"best_{model_name.lower()}.pth")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}")
              
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), best_model_path)
            
    print(f"Finished Training {model_name}. Best Val Acc: {best_val_acc:.4f}. Saved to {best_model_path}")
    return best_model_path

if __name__ == '__main__':
    print("Starting Mammography Deep Learning Training Pipeline...")
    jpeg_mapping = build_jpeg_mapping()
    
    train_dataset = MammographyDataset(SPLIT_CSV, split='train', jpeg_mapping=jpeg_mapping, transform=train_transform)
    val_dataset = MammographyDataset(SPLIT_CSV, split='validation', jpeg_mapping=jpeg_mapping, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    models_to_train = ['VGG16', 'ResNet50', 'EfficientNetB0']
    
    for m_name in models_to_train:
        train_and_evaluate(m_name, train_loader, val_loader)
        
    print("\nAll models trained successfully and checkpoints saved.")