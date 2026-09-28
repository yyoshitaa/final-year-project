import os
import csv
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = r"C:\Users\thera\Mammography_Breast_Cancer_Project"
SPLIT_CSV = os.path.join(BASE_DIR, "dataset_split.csv")
DICOM_INFO_CSV = r"C:\Mammography_Project\csv\dicom_info.csv"

def build_jpeg_mapping():
    mapping = {}
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
    return mapping

class MammographyTestDataset(Dataset):
    def __init__(self, csv_file, split='test', jpeg_mapping=None, transform=None):
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
            return img_tensor, label
            
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((224, 224), dtype=np.uint8)
            
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        if self.transform:
            img_tensor = self.transform(pil_img)
        else:
            img_tensor = transforms.ToTensor()(pil_img)
            
        return img_tensor, label

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_trained_model(model_name):
    if model_name == 'VGG16':
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
        path = os.path.join(BASE_DIR, "best_vgg16.pth")
    elif model_name == 'ResNet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        path = os.path.join(BASE_DIR, "best_resnet50.pth")
    elif model_name == 'EfficientNetB0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        path = os.path.join(BASE_DIR, "best_efficientnetb0.pth")
    else:
        raise ValueError("Unknown model")
        
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE)

def evaluate():
    print("Loading JPEG mapping for testing...")
    jpeg_mapping = build_jpeg_mapping()
    test_dataset = MammographyTestDataset(SPLIT_CSV, split='test', jpeg_mapping=jpeg_mapping, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    print(f"Test dataset size: {len(test_dataset)}")
    
    models_to_test = ['VGG16', 'ResNet50', 'EfficientNetB0']
    results = {}
    
    plt.figure(figsize=(8, 6))
    
    for m_name in models_to_test:
        print(f"\nEvaluating {m_name}...")
        model = load_trained_model(m_name)
        model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        
        # Sensitivity is Recall for positive class (malignant)
        sensitivity = recall
        # Specificity calculation from confusion matrix
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            plt.plot(fpr, tpr, label=f'{m_name} (AUC = {auc:.3f})')
        except Exception:
            auc = 0.0
            
        results[m_name] = {
            'Accuracy': acc,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Precision': precision,
            'F1': f1,
            'AUC': auc,
            'CM': cm
        }
        
        print(f"[{m_name}] Acc: {acc:.4f} | Sens: {sensitivity:.4f} | Spec: {specificity:.4f} | Prec: {precision:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    # Plot ROC curves
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(BASE_DIR, 'roc_curves.png'))
    plt.close()
    print("\nSaved ROC curves plot to roc_curves.png")
    
    print("\n================ MODEL COMPARISON TABLE ================")
    print(f"{'Model':<15} | {'Accuracy':<9} | {'Sensitivity':<11} | {'Specificity':<11} | {'Precision':<9} | {'F1-Score':<9} | {'AUC':<6}")
    print("-" * 80)
    for m_name, metrics in results.items():
        print(f"{m_name:<15} | {metrics['Accuracy']:<9.4f} | {metrics['Sensitivity']:<11.4f} | {metrics['Specificity']:<11.4f} | {metrics['Precision']:<9.4f} | {metrics['F1']:<9.4f} | {metrics['AUC']:<6.4f}")
    print("=" * 80)

if __name__ == '__main__':
    evaluate()