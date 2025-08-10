import nibabel as nib
import numpy as np
import glob

label_paths = glob.glob("data/ACDC/testing/patient101/patient101_frame01_gt.nii.gz")
all_labels = set()

for path in label_paths:
    seg = nib.load(path).get_fdata()
    all_labels.update(np.unique(seg).astype(int))

print("Unique label IDs:", sorted(all_labels))