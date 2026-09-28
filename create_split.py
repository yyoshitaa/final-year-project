import csv
import os
import random
from collections import defaultdict, Counter

BASE = r"C:\Mammography_Project"
CSV_DIR = os.path.join(BASE, "csv")
OUTPUT = os.path.join(
    r"C:\Users\thera\Mammography_Breast_Cancer_Project",
    "dataset_split.csv"
)


def load_dicom_info():
    mapping = {}

    path = os.path.join(CSV_DIR, "dicom_info.csv")

    with open(path, encoding="utf-8") as f:
        rows = csv.DictReader(f)

        for row in rows:
            patient_id = row["PatientID"].strip()
            description = row["SeriesDescription"].strip().lower()
            image_path = row["image_path"].strip()

            if not patient_id:
                continue

            if description != "full mammogram images":
                continue

            relative = image_path.replace(
                "CBIS-DDSM/jpeg/",
                ""
            )

            local_path = os.path.join(
                r"C:\Mammography_Project\jpeg",
                relative
            )

            if os.path.exists(local_path):
                mapping[patient_id] = local_path

    return mapping


def load_cases(filename):
    path = os.path.join(CSV_DIR, filename)

    cases = []

    with open(path, encoding="utf-8") as f:
        rows = csv.DictReader(f)

        for row in rows:

            pathology = row["pathology"].strip().upper()

            if pathology == "MALIGNANT":
                label = 1
            elif pathology in [
                "BENIGN",
                "BENIGN_WITHOUT_CALLBACK"
            ]:
                label = 0
            else:
                continue

            case_id = row["image file path"].split("/")[0]
            patient_id = row["patient_id"].strip()

            cases.append({
                "patient_id": patient_id,
                "case_id": case_id,
                "label": label,
                "pathology": pathology
            })

    return cases


# Load cases
mass = load_cases("mass_case_description_train_set.csv")
calc = load_cases("calc_case_description_train_set.csv")

cases = mass + calc

# Build patient groups
patients = defaultdict(list)

for case in cases:
    patients[case["patient_id"]].append(case)


# Determine patient-level label
patient_label = {}

for patient, patient_cases in patients.items():

    labels = [c["label"] for c in patient_cases]

    # If any case is malignant, treat patient as malignant.
    patient_label[patient] = 1 if 1 in labels else 0


# Separate patients by class
malignant_patients = [
    p for p, label in patient_label.items()
    if label == 1
]

benign_patients = [
    p for p, label in patient_label.items()
    if label == 0
]


random.seed(42)

random.shuffle(malignant_patients)
random.shuffle(benign_patients)


def split_group(group):

    n = len(group)

    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train = group[:n_train]
    val = group[n_train:n_train + n_val]
    test = group[n_train + n_val:]

    return train, val, test


m_train, m_val, m_test = split_group(malignant_patients)
b_train, b_val, b_test = split_group(benign_patients)


train_patients = set(m_train + b_train)
val_patients = set(m_val + b_val)
test_patients = set(m_test + b_test)


# Save case-level split
with open(OUTPUT, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "patient_id",
        "case_id",
        "label",
        "pathology",
        "split"
    ])

    for case in cases:

        patient = case["patient_id"]

        if patient in train_patients:
            split = "train"
        elif patient in val_patients:
            split = "validation"
        elif patient in test_patients:
            split = "test"
        else:
            continue

        writer.writerow([
            case["patient_id"],
            case["case_id"],
            case["label"],
            case["pathology"],
            split
        ])


print("========== PATIENT SPLIT ==========")

print("Total patients:", len(patient_label))

print("\nPatients:")
print("Train:", len(train_patients))
print("Validation:", len(val_patients))
print("Test:", len(test_patients))

print("\nImages:")

for split in ["train", "validation", "test"]:

    subset = [
        c for c in cases
        if (
            c["patient_id"] in
            (train_patients if split == "train"
             else val_patients if split == "validation"
             else test_patients)
        )
    ]

    print(split, ":", len(subset))

    print(
        "  malignant:",
        sum(c["label"] == 1 for c in subset)
    )

    print(
        "  benign:",
        sum(c["label"] == 0 for c in subset)
    )


print("\nSaved:", OUTPUT)
print("===================================")