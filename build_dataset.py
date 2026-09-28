import csv
import os
from collections import Counter

BASE = r"C:\Mammography_Project"
CSV_DIR = os.path.join(BASE, "csv")
JPEG_DIR = os.path.join(BASE, "jpeg")


def load_dicom_info():
    """
    Create a mapping:

    PatientID + image type
            ↓
    actual local JPEG
    """

    mapping = {}

    path = os.path.join(CSV_DIR, "dicom_info.csv")

    with open(path, encoding="utf-8") as f:
        rows = csv.DictReader(f)

        for row in rows:

            patient_id = row["PatientID"].strip()
            description = row["SeriesDescription"].strip().lower()
            image_path = row["image_path"].strip()

            if not patient_id or not image_path:
                continue

            # We only want full mammogram images.
            if description != "full mammogram images":
                continue

            relative_path = image_path.replace(
                "CBIS-DDSM/jpeg/",
                ""
            )

            local_path = os.path.join(
                JPEG_DIR,
                relative_path
            )

            if os.path.exists(local_path):

                # A patient/case can have multiple records,
                # so preserve all available full mammograms.
                mapping.setdefault(patient_id, []).append(
                    local_path
                )

    return mapping


def load_cases(filename, dicom_mapping):

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

            patient_id = row["patient_id"].strip()

            # Try to identify the exact case.
            case_id = row["image file path"].split("/")[0]

            image_candidates = []

            # Exact case match
            if case_id in dicom_mapping:
                image_candidates = dicom_mapping[case_id]

            # Fallback: patient ID
            elif patient_id in dicom_mapping:
                image_candidates = dicom_mapping[patient_id]

            for jpeg_path in image_candidates:

                cases.append({
                    "patient_id": patient_id,
                    "case_id": case_id,
                    "pathology": pathology,
                    "label": label,
                    "jpeg_path": jpeg_path
                })

    return cases


print("Loading DICOM information...")

dicom_mapping = load_dicom_info()

print("Full-mammogram case IDs:", len(dicom_mapping))


print("\nLoading mass cases...")

mass_cases = load_cases(
    "mass_case_description_train_set.csv",
    dicom_mapping
)


print("Loading calcification cases...")

calc_cases = load_cases(
    "calc_case_description_train_set.csv",
    dicom_mapping
)


cases = mass_cases + calc_cases


print("\n========== FINAL DATASET CHECK ==========")

print("Mass mapped cases:", len(mass_cases))
print("Calcification mapped cases:", len(calc_cases))
print("Total usable images:", len(cases))

print("\nPathology:")
print(Counter(c["pathology"] for c in cases))

print("\nBinary labels:")
print(Counter(c["label"] for c in cases))

print("\nUnique patients:")
print(len(set(c["patient_id"] for c in cases)))


print("\nFirst 10 images:")

for case in cases[:10]:

    print("\nPatient:", case["patient_id"])
    print("Case:", case["case_id"])
    print("Pathology:", case["pathology"])
    print("Label:", case["label"])
    print("JPEG:", case["jpeg_path"])


print("\n========================================")