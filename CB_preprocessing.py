import mne
import os
import glob
from mne.preprocessing import ICA
from mne_icalabel import label_components

data_path = "data"
preprocessed_path = "data/derivatives/preprocessed"
os.makedirs(preprocessed_path, exist_ok=True)

subjects = range(1, 30)
sessions = range(1, 4)
tasks_to_process = ['Flanker', 'PVT', 'twoBACK']

for sub_id in subjects:
    for ses_id in sessions:
        subject = f"sub-{sub_id:02d}"
        session = f"ses-S{ses_id}"
        eeg_path = os.path.join(data_path, subject, session, 'eeg')
        
        print(f"\nProcessing {subject}, {session}")

        task_files_for_ica_fit = []
        for task_name in tasks_to_process:
            file_path = os.path.join(eeg_path, f'{task_name}.set')
            if os.path.exists(file_path):
                task_files_for_ica_fit.append(file_path)

        raws_for_ica = []
        for file_path in task_files_for_ica_fit:
            raw = mne.io.read_raw_eeglab(file_path, preload=True)
            raw.filter(l_freq=1.0, h_freq=100, fir_design='firwin')
            raws_for_ica.append(raw)
        
        raw_concatenated = mne.concatenate_raws(raws_for_ica)

        if 'Cz' in raw_concatenated.ch_names:
            raw_concatenated.drop_channels(['Cz'])
        
        channel_types = {ch: 'eeg' for ch in raw_concatenated.ch_names if ch != 'ECG1'}
        if 'ECG1' in raw_concatenated.ch_names:
            channel_types['ECG1'] = 'ecg'
        raw_concatenated.set_channel_types(channel_types)
        
        raw_concatenated.set_eeg_reference('average')

        ica = ICA(n_components=0.99, max_iter='auto', method='infomax', 
                  fit_params=dict(extended=True), random_state=715)

        ica.fit(raw_concatenated)

        ic_labels = label_components(raw_concatenated, ica, method='iclabel')
        labels = ic_labels["labels"]
        ica.exclude = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
        print(f"Components Labels: {ica.labels_}")

        for task_name in tasks_to_process:
            task_file_path = os.path.join(eeg_path, f'{task_name}.set')

            raw_task = mne.io.read_raw_eeglab(task_file_path, preload=True)
            
            raw_task.filter(l_freq=1.0, h_freq=100, fir_design='firwin')

            if 'Cz' in raw_task.ch_names:
                raw_task.drop_channels(['Cz'])
            raw_task.set_channel_types(channel_types)
            raw_task.set_eeg_reference('average')

            ica.apply(raw_task)

            raw_task.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin')
            raw_task.notch_filter(freqs=50.0, fir_design='firwin')

            raw_task.set_eeg_reference('average', projection=True)

            save_path = os.path.join(preprocessed_path, subject, session, 'eeg')
            os.makedirs(save_path, exist_ok=True)
            save_fname = f"{task_name}_preprocessed.fif"
            full_save_path = os.path.join(save_path, save_fname)
            raw_task.save(full_save_path, overwrite=True)
