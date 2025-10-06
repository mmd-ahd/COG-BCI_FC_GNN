import mne
import os
import glob

preprocessed_path = "data/derivatives/preprocessed"
epochs_path = "data/derivatives/epochs"
subjects = range(1, 30)
sessions = range(1, 4)

event_ids_for_epoching = {
    'twoBACK': {'Normal': 6221, 'Hit': 6222},
    'Flanker': {'Congruent': 241, 'Incongruent': 242},
    'PVT': {'Stimulus': 13}
}

event_map = {str(code): code for task in event_ids_for_epoching for code in event_ids_for_epoching[task].values()}

tmin, tmax = -0.2, 1
baseline = (-0.2, 0)
reject = dict(eeg=100e-6)

os.makedirs(epochs_path, exist_ok=True)

for sub_id in subjects:
    subject = f"sub-{sub_id:02d}"
    for ses_id in sessions:
        session = f"ses-S{ses_id}"
        session_bids = f"ses-{ses_id:02d}"
        eeg_path = os.path.join(preprocessed_path, subject, session, 'eeg')

        for task_name in event_ids_for_epoching.keys():
            
            file_name = f"{task_name}_preprocessed.fif"
            file_path = os.path.join(eeg_path, file_name)

            if os.path.exists(file_path):
                raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)

                events, _ = mne.events_from_annotations(raw, event_id=event_map, verbose=False)

                epochs = mne.Epochs(raw, events, event_id=event_ids_for_epoching[task_name],
                                    tmin=tmin, tmax=tmax, baseline=baseline, reject=reject,
                                    preload=True, proj=True)
                
                if len(epochs) > 0:
                    save_dir = os.path.join(epochs_path, subject, session)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    save_fname = f"{subject}_{session_bids}_{task_name}-epo.fif"
                    full_save_path = os.path.join(save_dir, save_fname)
                    
                    epochs.save(full_save_path, overwrite=True)
                    print(f"Created and saved {len(epochs)} epochs for {task_name} to {save_fname}")