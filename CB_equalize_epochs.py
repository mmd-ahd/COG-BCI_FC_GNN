import mne
import os
import glob
import numpy as np

epochs_path = "data/derivatives/epochs"

all_epoch_files = glob.glob(os.path.join(epochs_path, "**", "*-epo.fif"), recursive=True)

min_epoch_count = float('inf')

for fpath in all_epoch_files:

    epochs = mne.read_epochs(fpath, preload=False)
    if len(epochs) < min_epoch_count:
        min_epoch_count = len(epochs)
        
print(f"Minimum epoch count found: {min_epoch_count}")

for fpath in all_epoch_files:

    epochs = mne.read_epochs(fpath, preload=True, proj=True)
    
    if len(epochs) > min_epoch_count:
        gfp_std = epochs.get_data().std(axis=1).std(axis=1)
        
        indices_to_keep = np.argsort(gfp_std)[:min_epoch_count]
        
        epochs_to_keep = epochs[indices_to_keep]

        epochs_to_keep.events[:, 0].sort()
        
        epochs_to_keep.save(fpath, overwrite=True)
        print(f"Processed {os.path.basename(fpath)}: kept {min_epoch_count} epochs.")
    else:
        print(f"Skipped {os.path.basename(fpath)}: already has {len(epochs)} epochs.")
