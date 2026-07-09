import os
import shutil
import nibabel as nib

# === Please set your own paths below! ===
img_root = "/home/yang/Downloads/Projects/TextBraTS_4chan/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
txt_root = "/home/yang/Downloads/Projects/TextBraTS_4chan/data/TextBraTSData"
out_root = "/home/yang/Downloads/Projects/TextBraTS_4chan/code/data/TextBraTSData"

# Loop over all cases in the image folder
for case in os.listdir(img_root):
    img_case_dir = os.path.join(img_root, case)
    txt_case_dir = os.path.join(txt_root, case)
    out_case_dir = os.path.join(out_root, case)

    if not os.path.isdir(img_case_dir):
        continue  # Skip non-directory files

    # Create output folder for each case
    os.makedirs(out_case_dir, exist_ok=True)

    # Copy all imaging files and segmentation labels
    for file in os.listdir(img_case_dir):
        src_path = os.path.join(img_case_dir, file)

        # Handle .nii files: convert to .nii.gz
        if file.endswith(".nii") and not file.endswith(".nii.gz"):
            img = nib.load(src_path)
            new_file_name = file + ".gz"
            dest_path = os.path.join(out_case_dir, new_file_name)
            nib.save(img, dest_path)
        else:
            # Just copy everything else (including .nii.gz)
            dest_path = os.path.join(out_case_dir, file)
            shutil.copy2(src_path, dest_path)
    # for file in os.listdir(img_case_dir):
    #     shutil.copy2(os.path.join(img_case_dir, file), os.path.join(out_case_dir, file))

    # Copy text reports and feature files if available
    if os.path.exists(txt_case_dir):
        for file in os.listdir(txt_case_dir):
            shutil.copy2(os.path.join(txt_case_dir, file), os.path.join(out_case_dir, file))
    else:
        print(f"Warning: {txt_case_dir} does not exist, skipping.")

print("Merge done! All cases are in:", out_root)
