import os
import mne
import mne_connectivity
import numpy as np

subjects = range(20, 30)
sessions = range(1, 4)

epochs_base_path = "data/derivatives/epochs"
connectivity_base_path = "data/derivatives/connectivity"
os.makedirs(connectivity_base_path, exist_ok=True)

tasks = ['twoBACK', 'Flanker', 'PVT']
fmin = (4, 8, 13, 30)
fmax = (8, 13, 30, 45)
freqs = np.arange(4, 46)
n_cycles = freqs * 3/4
method = 'wpli'
mode = 'cwt_morlet'

for sub_id in subjects:
    
    subject = f"sub-{sub_id:02d}"
    
    for ses_id in sessions:
        
        session = f"ses-S{ses_id}"
        session_bids = f"ses-{ses_id:02d}"
        epochs_dir = os.path.join(epochs_base_path, subject, session)
        connectivity_dir = os.path.join(connectivity_base_path, subject, session)
        os.makedirs(connectivity_dir, exist_ok=True)
        
        for task in tasks:
            
            epochs_file = f"{subject}_{session_bids}_{task}-epo.fif"
            epochs_file_path = os.path.join(epochs_dir, epochs_file)
            epochs = mne.read_epochs(epochs_file_path, preload=True)
            
            con = mne_connectivity.spectral_connectivity_epochs(
                epochs,
                names=epochs.ch_names,
                method=method,
                mode=mode,
                fmin=fmin,
                fmax=fmax,
                faverage=True,
                cwt_freqs=freqs,
                cwt_n_cycles=n_cycles
            )
            
            connectivity_name = f"{subject}_{session_bids}_{task}_connectivity.nc"
            connectivity_file_path = os.path.join(connectivity_dir, connectivity_name)
            con.save(connectivity_file_path)
