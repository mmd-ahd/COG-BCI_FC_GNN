import os
import numpy as np
import mne
import mne_connectivity
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import Dict, List, Tuple


class EEGGraphDataset(Dataset):
    
    def __init__(
        self,
        epochs_root: str,
        connectivity_root: str,
        subjects: List[int],
        sessions: List[int],
        tasks: List[str],
        label_mapping: Dict[str, int],
        k_neighbors: int = 10,
        window_size_ms: float = 200.0,
        step_size_ms: float = 40.0,
        normalize_data: bool = True
    ):
        super().__init__()
        
        self.epochs_root = epochs_root
        self.connectivity_root = connectivity_root
        self.subjects = subjects
        self.sessions = sessions
        self.tasks = tasks
        self.label_mapping = label_mapping
        self.k_neighbors = k_neighbors
        self.window_size_ms = window_size_ms
        self.step_size_ms = step_size_ms
        self.normalize_data = normalize_data
        
        self.data_catalog = []
        self.connectivity_cache = {}
        
        self._build_catalog()
        
        print(f"Dataset initialized with {len(self.data_catalog)} total epochs")
        print(f"Label distribution: {self._get_label_distribution()}")
        print(f"Data normalization: {'ENABLED' if normalize_data else 'DISABLED'}")
    
    def _build_catalog(self):
        for sub_id in self.subjects:
            subject = f"sub-{sub_id:02d}"
            
            for ses_id in self.sessions:
                session = f"ses-S{ses_id}"
                session_bids = f"ses-{ses_id:02d}"
                
                for task in self.tasks:
                    epoch_file = f"{subject}_{session_bids}_{task}-epo.fif"
                    epoch_path = os.path.join(self.epochs_root, subject, session, epoch_file)
                    
                    connectivity_file = f"{subject}_{session_bids}_{task}_connectivity.nc"
                    connectivity_path = os.path.join(self.connectivity_root, subject, session, connectivity_file)
                    
                    if not os.path.exists(epoch_path) or not os.path.exists(connectivity_path):
                        continue
                    
                    if task not in self.label_mapping:
                        continue
                    
                    label = self.label_mapping[task]
                    
                    try:
                        epochs_obj = mne.read_epochs(epoch_path, preload=False, verbose=False)
                        n_epochs = len(epochs_obj)
                        
                        for epoch_idx in range(n_epochs):
                            self.data_catalog.append({
                                'subject': subject,
                                'session': session,
                                'task': task,
                                'label': label,
                                'epoch_file': epoch_path,
                                'connectivity_file': connectivity_path,
                                'epoch_index': epoch_idx,
                                'session_key': f"{subject}_{session_bids}_{task}"
                            })
                    except Exception as e:
                        print(f"Error loading {epoch_path}: {e}")
                        continue
    
    def _get_label_distribution(self) -> Dict[int, int]:
        label_counts = {}
        for item in self.data_catalog:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts
    
    def _load_connectivity(self, connectivity_path: str) -> np.ndarray:
        if connectivity_path in self.connectivity_cache:
            return self.connectivity_cache[connectivity_path]
        
        conn = mne_connectivity.read_connectivity(connectivity_path)
        conn_array = conn.get_data(output='dense')
        
        if conn_array.ndim == 5:
            conn_array = conn_array[0]
        
        self.connectivity_cache[connectivity_path] = conn_array
        return conn_array
    
    def _get_window_connectivity(
        self, 
        connectivity: np.ndarray, 
        window_start_sample: int,
        window_end_sample: int
    ) -> np.ndarray:
        conn_window = connectivity[:, :, :, window_start_sample:window_end_sample]
        conn_time_avg = np.mean(conn_window, axis=3)
        conn_matrix = np.mean(conn_time_avg, axis=2)
        return conn_matrix
    
    def _sparsify_graph(self, adj_matrix: np.ndarray) -> torch.Tensor:
        num_nodes = adj_matrix.shape[0]
        k = min(self.k_neighbors, num_nodes - 1)
        
        adj_tensor = torch.from_numpy(adj_matrix).float()
        adj_tensor.fill_diagonal_(-float('inf'))
        
        topk_values, topk_indices = torch.topk(adj_tensor, k=k, dim=1, largest=True)
        
        source_nodes = torch.arange(num_nodes).unsqueeze(1).expand(-1, k).flatten()
        target_nodes = topk_indices.flatten()
        
        edge_index = torch.stack([source_nodes, target_nodes], dim=0).long()
        
        return edge_index
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        mean = np.mean(signal, axis=1, keepdims=True)
        std = np.std(signal, axis=1, keepdims=True)
        
        std = np.where(std == 0, 1.0, std)
        
        normalized = (signal - mean) / std
        
        return normalized
    
    def _create_windows(
        self, 
        epoch_data: np.ndarray, 
        connectivity: np.ndarray, 
        sfreq: float
    ) -> List[Data]:
        num_samples = epoch_data.shape[1]
        
        if self.normalize_data:
            epoch_data = self._normalize_signal(epoch_data)
        
        window_size_samples = int(self.window_size_ms * sfreq / 1000.0)
        step_size_samples = int(self.step_size_ms * sfreq / 1000.0)
        
        data_list = []
        
        window_start = 0
        while window_start + window_size_samples <= num_samples:
            window_end = window_start + window_size_samples
            
            signal_window = epoch_data[:, window_start:window_end]
            
            x = torch.from_numpy(signal_window).float().unsqueeze(1)
            
            conn_matrix = self._get_window_connectivity(
                connectivity, 
                window_start, 
                window_end
            )
            
            edge_index = self._sparsify_graph(conn_matrix)
            
            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)
            
            window_start += step_size_samples
        
        return data_list
    
    def __len__(self) -> int:
        return len(self.data_catalog)
    
    def __getitem__(self, idx: int) -> Tuple[List[Data], int, str]:
        metadata = self.data_catalog[idx]
        epoch_file = metadata['epoch_file']
        connectivity_file = metadata['connectivity_file']
        epoch_idx = metadata['epoch_index']
        label = metadata['label']
        session_key = metadata['session_key']
        
        epochs_obj = mne.read_epochs(epoch_file, preload=True, verbose=False)
        single_epoch = epochs_obj[epoch_idx]
        
        epoch_data = single_epoch.get_data()[0]
        sfreq = single_epoch.info['sfreq']
        
        del epochs_obj, single_epoch
        
        connectivity = self._load_connectivity(connectivity_file)
        data_sequence = self._create_windows(epoch_data, connectivity, sfreq)
        
        del epoch_data
        
        return data_sequence, label, session_key
    
    def get_session_groups(self) -> Dict[str, List[int]]:
        session_groups = {}
        for idx, metadata in enumerate(self.data_catalog):
            session_key = metadata['session_key']
            if session_key not in session_groups:
                session_groups[session_key] = []
            session_groups[session_key].append(idx)
        return session_groups
    
    def get_subject_sessions(self) -> Dict[str, List[str]]:
        subject_sessions = {}
        for metadata in self.data_catalog:
            subject = metadata['subject']
            session_key = metadata['session_key']
            if subject not in subject_sessions:
                subject_sessions[subject] = []
            if session_key not in subject_sessions[subject]:
                subject_sessions[subject].append(session_key)
        return subject_sessions
    
    def clear_cache(self):
        self.connectivity_cache.clear()
        print(f"Cleared connectivity cache")