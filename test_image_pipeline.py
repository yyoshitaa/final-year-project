import csv
import os

import cv2
import numpy as np
import torch

BASE = r"C:\Mammography_Project"
SPLIT_CSV = r"C:\Users\thera\Mammography_Breast_Cancer_Project\dataset_split.csv"
DICOM_CSV = os.path.join(BASE, "csv", "dicom_info.csv")

# --------------------------------------------------
# Build exact CASE ID -> JPEG mapping
# --------------------------------------------------

jpeg_map = {}

with open(DICOM_CSV, encoding="utf-8") as f:
    rows = csv.DictReader(f)

    for row in rows:
        case_id = row["PatientID"].strip()
        description = row["SeriesDescription"].strip().lower()
        image_path = row["image_path"].strip()

        if description != "full mammogram images":
            continue

        relative_path = image_path.replace("CBIS-DDSM/jpeg/", "")
        local_path = os.path.join(BASE, "jpeg", relative_path)

        if os.path.exists(local_path):
            jpeg_map[case_id] = local_path


print("Full mammogram mappings:", len(jpeg_map))


# --------------------------------------------------
# Select one training image
# --------------------------------------------------

selected_case = None
selected_patient = None
selected_label = None

with open(SPLIT_CSV, encoding="utf-8") as f:
    rows = csv.DictReader(f)

    for row in rows:
        if row["split"] == "train":
            selected_patient = row["patient_id"]
            selected_case = row["case_id"]
            selected_label = row["label"]
            break


print("\nPatient:", selected_patient)
print("Case:", selected_case)
print("Label:", selected_label)


# --------------------------------------------------
# Find JPEG
# --------------------------------------------------

image_path = jpeg_map.get(selected_case)

if image_path is None:
    raise RuntimeError(
        f"Could not find JPEG for case: {selected_case}"
    )

print("Image:")
print(image_path)


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise RuntimeError("OpenCV could not read the image.")


print("\nOriginal image:")
print("Shape:", image.shape)
print("Min:", image.min())
print("Max:", image.max())


# --------------------------------------------------
# CLAHE
# --------------------------------------------------

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

image = clahe.apply(image)


# --------------------------------------------------
# Resize
# --------------------------------------------------

image = cv2.resize(
    image,
    (224, 224),
    interpolation=cv2.INTER_AREA
)


# --------------------------------------------------
# Grayscale -> 3 channels
# --------------------------------------------------

image = np.stack(
    [image, image, image],
    axis=0
)


# --------------------------------------------------
# Normalize 0-1
# --------------------------------------------------

image = image.astype(np.float32) / 255.0


# --------------------------------------------------
# PyTorch tensor
# --------------------------------------------------

tensor = torch.from_numpy(image)

# [3,224,224] -> [1,3,224,224]
tensor = tensor.unsqueeze(0)


print("\nTensor:")
print("Shape:", tensor.shape)
print("Dtype:", tensor.dtype)
print("Min:", tensor.min().item())
print("Max:", tensor.max().item())


# --------------------------------------------------
# GPU
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    tensor = tensor.to(device)

    print(
        "Tensor device:",
        tensor.device
    )


print("\nIMAGE PIPELINE TEST SUCCESSFUL!")