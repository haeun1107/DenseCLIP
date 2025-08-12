import os
import numpy as np
from scipy.sparse import load_npz

label_dir = "data/BTCV/label"
all_labels = set()

for fname in os.listdir(label_dir):
    if fname.endswith(".npz"):
        M = load_npz(os.path.join(label_dir, fname)).toarray()
        if M.shape[0] == 13:
            seg = M.reshape(13, 512, 512).argmax(axis=0)
        elif M.shape[1] == 13:
            seg = M.T.reshape(13, 512, 512).argmax(axis=0)
        else:
            seg = M.reshape(512, 512)
        all_labels.update(np.unique(seg))

print("All BTCV labels:", sorted(all_labels))

import nibabel as nib

path = "data/ACDC/training/patient001/patient001_frame01_gt.nii.gz"
nii = nib.load(path)
label_data = np.asanyarray(nii.dataobj)

unique_vals = np.unique(label_data)
print("All ACDC labels:", unique_vals)