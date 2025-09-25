import mne
import os
import glob
from mne.preprocessing import ICA
from mne_icalabel import label_components

data_path = r"data"
preprocessed_path = r"data/derivatives/preprocessed"

os.makedirs(preprocessed_path, exist_ok=True)

subjects = range(1, 30)
sessions = range(1, 4)

for sub_id in subjects:
    for ses_id in sessions:
        subject = f"sub-{sub_id:02d}"
        session = f"ses-S{ses_id}"
        eeg_path = os.path.join(data_path, subject, session, 'eeg')
            
        raw_files = glob.glob(os.path.join(eeg_path, '*.set'))
        print(f"\nProcessing {subject}, {session}")
        
        for file_path in raw_files:
            raw = mne.io.read_raw_eeglab(file_path, preload=True)
            base_fname = os.path.basename(file_path)
            file_name_no_ext = os.path.splitext(base_fname)[0]
            print(f"  Processing file: {base_fname}")

            channel_types = {ch: 'eeg' for ch in raw.ch_names if ch != 'ECG1'}
            channel_types['ECG1'] = 'ecg'
            raw.set_channel_types(channel_types)

            raw.filter(l_freq=1, h_freq=100.0, fir_design='firwin')
            
            raw.set_eeg_reference('average')
            
            ica = ICA(max_iter='auto', method='infomax', fit_params=dict(extended=True), n_components=20, random_state=715)
            ica.fit(raw)

            ic_labels = label_components(raw, ica, method='iclabel')

            labels = ic_labels["labels"]
            exclude_idx = [
                idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
            ]
            ica.exclude = exclude_idx
            print(f"     ICLabel found: {ica.labels_}")

            ica.apply(raw, exclude=ica.exclude)            
            
            raw.filter(l_freq=None, h_freq=40.0, fir_design='firwin')
            raw.notch_filter(freqs=50.0, fir_design='firwin')

            raw.set_eeg_reference('average', projection=True)
            
            save_path = os.path.join(preprocessed_path, subject, session, 'eeg')
            os.makedirs(save_path, exist_ok=True)
            save_fname = f"{file_name_no_ext}_preprocessed.fif"
            full_save_path = os.path.join(save_path, save_fname)
            raw.save(full_save_path, overwrite=True)
            print(f"     Saved to {full_save_path}")
