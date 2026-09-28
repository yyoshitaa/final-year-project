import csv
import os

csv_file = r"C:\Mammography_Project\csv\dicom_info.csv"
jpeg_root = r"C:\Mammography_Project\jpeg"

total = 0
found = 0
samples = []

with open(csv_file, encoding="utf-8") as f:
    rows = csv.DictReader(f)

    for row in rows:
        total += 1

        image_path = row["image_path"]

        # Remove the dataset prefix
        relative_path = image_path.replace("CBIS-DDSM/jpeg/", "")

        # Build the real local path
        full_path = os.path.join(jpeg_root, relative_path)

        if os.path.exists(full_path):
            found += 1

            if len(samples) < 5:
                samples.append(
                    (
                        row["PatientID"],
                        row["SeriesDescription"],
                        full_path,
                    )
                )

print("Total dicom_info rows:", total)
print("JPEGs found:", found)
print("JPEGs missing:", total - found)

print("\nSample mappings:")

for sample in samples:
    print("\nPatient:", sample[0])
    print("Description:", sample[1])
    print("JPEG:", sample[2])