import mne
import os
import glob
import numpy as np

epochs_path = "data/derivatives/epochs"

all_epoch_files = glob.glob(os.path.join(epochs_path, "**", "**", "*-epo.fif"), recursive=True)

maximum_epoch_count = 80

for fpath in all_epoch_files:

    epochs = mne.read_epochs(fpath, preload=True, proj=True)
    
    if epochs.__len__() > maximum_epoch_count:
        gfp_std = epochs.get_data().std(axis=1).std(axis=1)
        
        indices_to_keep = np.argsort(gfp_std)[:maximum_epoch_count]
        
        num_epochs_kept = len(indices_to_keep)
        
        epochs_to_keep = epochs[indices_to_keep]

        epochs_to_keep.events[:, 0].sort()
        
        epochs_to_keep.save(fpath, overwrite=True)
        print(f"Processed {os.path.basename(fpath)}: kept {num_epochs_kept} epochs.")
    else:
        print(f"{os.path.basename(fpath)}: already has {len(epochs)} epochs.")
