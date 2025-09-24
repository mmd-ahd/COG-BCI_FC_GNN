import mne
import os

example_sub = "data\sub-01\ses-S1\eeg"
file_name = "Flanker.set"
full_path = os.path.join(example_sub, file_name)

raw = mne.io.read_raw_eeglab(full_path, preload=True)

print(raw.info)
print(raw.ch_names)

raw.plot(
    duration=10,
    n_channels=len(raw.ch_names),
    scalings='auto',
    show=True,
    block=True
)

raw.compute_psd(fmin=0.5, fmax=40, picks='eeg').plot()
input()
