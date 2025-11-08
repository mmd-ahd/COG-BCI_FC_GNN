import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import defaultdict
from tqdm import tqdm

from CB_Dataset import EEGGraphDataset
from CB_DCGATT import DCGATT

K_NEIGHBORS = 12
WINDOW_SIZE_MS = 200.0
STEP_SIZE_MS = 40.0

CNN_PARAMS = {
    'in_channels': 1,
    'out_channels': 48,
    'kernel_size': 5,
    'negative_slope': 0.01
}

GAT_PARAMS = {
    'hidden_channels': 64,
    'out_channels': 96,
    'heads': 8,
    'dropout': 0.235
}

TRANSFORMER_PARAMS = {
    'num_layers': 1,
    'num_heads': 8,
    'hidden_dim': 128,
    'dropout': 0.391
}

POOLING_PARAMS = {
    'spatial_gate_hidden_dim': 96,
    'temporal_attention_hidden_dim': 96
}

MLP_PARAMS = {
    'hidden_dim': 64,
    'dropout': 0.381
}

BATCH_SIZE = 16
ACCUMULATION_STEPS = 1
NUM_EPOCHS = 50
LEARNING_RATE = 0.000386
WEIGHT_DECAY = 0.000014
PATIENCE = 10
N_FOLDS = 5

USE_AMP = True
LABEL_SMOOTHING = 0.1
USE_COSINE_SCHEDULE = True
WARMUP_EPOCHS = 3

NUM_WORKERS = 4
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True

TARGET_PARAMS = 400000

ALTERNATIVE_CONFIG = {
    'cnn_out': 48,
    'cnn_kernel': 7,
    'gat_hidden': 48,
    'gat_out': 96,
    'gat_heads': 8,
    'gat_dropout': 0.36,
    'transformer_layers': 1,
    'transformer_heads': 8,
    'transformer_hidden': 256,
    'transformer_dropout': 0.36,
    'pooling_hidden': 64,
    'mlp_hidden': 96,
    'mlp_dropout': 0.32,
    'batch_size': 16,
    'learning_rate': 0.0005,
    'weight_decay': 1.18e-05,
    'k_neighbors': 19
}


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-7):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        
        return max(min_lr / optimizer.defaults['lr'], cosine_decay)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def collate_fn(batch):
    data_sequences, labels, session_keys = [], [], []
    
    for data_seq, label, session_key in batch:
        data_sequences.append(data_seq)
        labels.append(label)
        session_keys.append(session_key)
    
    seq_len = len(data_sequences[0])
    batched_sequences = []
    
    for t in range(seq_len):
        data_list_t = [data_sequences[b][t] for b in range(len(data_sequences))]
        batch_t = Batch.from_data_list(data_list_t)
        batched_sequences.append(batch_t)
    
    labels = torch.tensor(labels, dtype=torch.long)
    
    return batched_sequences, labels, session_keys


def process_batch_for_model(batched_sequences, device):
    x_list, edge_index_list = [], []
    
    for batch_t in batched_sequences:
        batch_t = batch_t.to(device)
        x_list.append(batch_t.x)
        edge_index_list.append(batch_t.edge_index)
    
    return x_list, edge_index_list


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, 
                accumulation_steps=1, use_amp=False):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    scaler = torch.amp.GradScaler('cuda') if use_amp and torch.cuda.is_available() else None
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, (batched_sequences, labels, _) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        x_list, edge_index_list = process_batch_for_model(batched_sequences, device)
        labels = labels.to(device)
        
        if use_amp and scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(x_list, edge_index_list)
                loss = criterion(logits, labels)
                loss = loss / accumulation_steps
        else:
            logits = model(x_list, edge_index_list)
            loss = criterion(logits, labels)
            loss = loss / accumulation_steps
        
        if torch.isnan(loss):
            print(f"WARNING: NaN loss at batch {batch_idx}")
            continue
        
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            
            if scheduler is not None and USE_COSINE_SCHEDULE:
                scheduler.step()
        
        total_loss += loss.item() * labels.size(0) * accumulation_steps
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        del logits, loss, x_list, edge_index_list, batched_sequences
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate_with_voting(model, dataloader, criterion, device, use_amp=False):
    model.eval()
    session_logits = defaultdict(list)
    session_labels = {}
    total_loss = 0.0
    num_epochs = 0
    
    with torch.no_grad():
        for batched_sequences, labels, session_keys in tqdm(dataloader, desc="Validation", leave=False):
            x_list, edge_index_list = process_batch_for_model(batched_sequences, device)
            labels = labels.to(device)
            
            if use_amp and torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    logits = model(x_list, edge_index_list)
                    loss = criterion(logits, labels)
            else:
                logits = model(x_list, edge_index_list)
                loss = criterion(logits, labels)
            
            total_loss += loss.item() * labels.size(0)
            num_epochs += labels.size(0)
            
            logits_np = logits.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            for i, session_key in enumerate(session_keys):
                session_logits[session_key].append(logits_np[i])
                session_labels[session_key] = labels_np[i]
            
            del logits, loss, x_list, edge_index_list, batched_sequences
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    session_preds, session_true = [], []
    for session_key in session_logits.keys():
        avg_logits = np.mean(session_logits[session_key], axis=0)
        pred = np.argmax(avg_logits)
        
        session_preds.append(pred)
        session_true.append(session_labels[session_key])
    
    avg_loss = total_loss / num_epochs
    session_accuracy = accuracy_score(session_true, session_preds)
    
    return avg_loss, session_accuracy


def test_with_voting(model, dataloader, device, use_amp=False):
    model.eval()
    session_logits = defaultdict(list)
    session_labels = {}
    
    with torch.no_grad():
        for batched_sequences, labels, session_keys in tqdm(dataloader, desc="Testing", leave=False):
            x_list, edge_index_list = process_batch_for_model(batched_sequences, device)
            
            if use_amp and torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    logits = model(x_list, edge_index_list)
            else:
                logits = model(x_list, edge_index_list)
            
            logits_np = logits.cpu().numpy()
            labels_np = labels.numpy()
            
            for i, session_key in enumerate(session_keys):
                session_logits[session_key].append(logits_np[i])
                session_labels[session_key] = labels_np[i]
            
            del logits, x_list, edge_index_list, batched_sequences
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    session_preds, session_true = [], []
    
    for session_key in session_logits.keys():
        avg_logits = np.mean(session_logits[session_key], axis=0)
        pred = np.argmax(avg_logits)
        
        session_preds.append(pred)
        session_true.append(session_labels[session_key])
    
    accuracy = accuracy_score(session_true, session_preds)
    conf_matrix = confusion_matrix(session_true, session_preds)
    
    return accuracy, conf_matrix, session_true, session_preds


def create_subject_splits(dataset, n_folds=5):
    subject_sessions = dataset.get_subject_sessions()
    subjects = list(subject_sessions.keys())
    session_groups = dataset.get_session_groups()
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    splits = []
    for fold_idx, (trainval_subjects_idx, test_subjects_idx) in enumerate(kfold.split(subjects)):
        trainval_subjects = [subjects[i] for i in trainval_subjects_idx]
        test_subjects = [subjects[i] for i in test_subjects_idx]
        
        np.random.seed(42 + fold_idx)
        np.random.shuffle(trainval_subjects)
        
        n_train = int(0.75 * len(trainval_subjects))
        train_subjects = trainval_subjects[:n_train]
        val_subjects = trainval_subjects[n_train:]
        
        train_indices, val_indices, test_indices = [], [], []
        
        for subject in train_subjects:
            for session_key in subject_sessions[subject]:
                train_indices.extend(session_groups[session_key])
        
        for subject in val_subjects:
            for session_key in subject_sessions[subject]:
                val_indices.extend(session_groups[session_key])
        
        for subject in test_subjects:
            for session_key in subject_sessions[subject]:
                test_indices.extend(session_groups[session_key])
        
        splits.append((train_indices, val_indices, test_indices))
        
        print(f"Fold {len(splits)}: Train={len(train_subjects)} subjects ({len(train_indices)} epochs), "
              f"Val={len(val_subjects)} subjects ({len(val_indices)} epochs), "
              f"Test={len(test_subjects)} subjects ({len(test_indices)} epochs)")
    
    return splits


def train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler,
               device, num_epochs, patience, fold_num, save_dir, 
               accumulation_steps, use_amp):
    early_stopping = EarlyStopping(patience=patience, mode='min')
    best_model_state = None
    best_val_loss = float('inf')
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("TRAINING CONFIGURATION:")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Batch size: {train_loader.batch_size}  Effective batch: {train_loader.batch_size * accumulation_steps}")
    print(f"  Learning rate: {LEARNING_RATE:.6f}  Weight decay: {WEIGHT_DECAY:.6f}  k_neighbors: {K_NEIGHBORS}")
    print(f"  Mixed precision: {use_amp}")
    
    for epoch in range(num_epochs):
        print(f"Fold {fold_num} - Epoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            accumulation_steps, use_amp
        )
        
        val_loss, val_acc = validate_with_voting(model, val_loader, criterion, device, use_amp)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"LR: {current_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc (session):   {val_acc:.4f}")
        
        acc_gap = train_acc - val_acc
        loss_gap = val_loss - train_loss
        print(f"Gaps: Acc={acc_gap:+.4f}, Loss={loss_gap:+.4f}", end="")
        
        if acc_gap > 0.10:
            print(" Overfitting detected")
        elif loss_gap < -0.05:
            print(" Generalizing well")
        else:
            print()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, os.path.join(save_dir, f'best_model_fold_{fold_num}.pt'))
            print(f"Best model saved (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f})")
        
        if early_stopping(val_loss, epoch):
            print(f"Early stopping at epoch {epoch+1}")
            print(f"Best was epoch {early_stopping.best_epoch+1} with val loss {early_stopping.best_score:.4f}")
            break
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return best_model_state, history


def main():
    EPOCHS_ROOT = "data/derivatives/epochs"
    CONNECTIVITY_ROOT = "data/derivatives/connectivity"
    SAVE_DIR = "results/dcgatt_optimized"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    config = {
        'k_neighbors': K_NEIGHBORS,
        'cnn_params': CNN_PARAMS,
        'gat_params': GAT_PARAMS,
        'transformer_params': TRANSFORMER_PARAMS,
        'pooling_params': POOLING_PARAMS,
        'mlp_params': MLP_PARAMS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'label_smoothing': LABEL_SMOOTHING,
        'use_cosine_schedule': USE_COSINE_SCHEDULE,
        'warmup_epochs': WARMUP_EPOCHS,
        'validation_mode': 'session-level',
        'early_stopping_metric': 'validation_loss'
    }
    
    with open(os.path.join(SAVE_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("OPTIMIZED DC-GATT TRAINING")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    print("Creating dataset...")
    dataset = EEGGraphDataset(
        epochs_root=EPOCHS_ROOT,
        connectivity_root=CONNECTIVITY_ROOT,
        subjects=list(range(1, 30)),
        sessions=list(range(1, 4)),
        tasks=['twoBACK', 'Flanker', 'PVT'],
        label_mapping={'twoBACK': 0, 'Flanker': 1, 'PVT': 2},
        k_neighbors=K_NEIGHBORS,
        normalize_data=True
    )
    
    sample_data, _, _ = dataset[0]
    NUM_NODES = sample_data[0].x.shape[0]
    print(f"Detected NUM_NODES: {NUM_NODES}")
    
    print("Creating subject-level 60/20/20 splits...")
    splits = create_subject_splits(dataset, n_folds=N_FOLDS)
    
    fold_results = []
    
    for fold_num, (train_indices, val_indices, test_indices) in enumerate(splits, 1):
        print(f"FOLD {fold_num}/{N_FOLDS}")
        
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, collate_fn=collate_fn,
            pin_memory=True, prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=PERSISTENT_WORKERS
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, collate_fn=collate_fn,
            pin_memory=True, prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=PERSISTENT_WORKERS
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, collate_fn=collate_fn,
            pin_memory=True, prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=PERSISTENT_WORKERS
        )
        
        model = DCGATT(
            num_nodes=NUM_NODES, seq_len=26, num_classes=3,
            cnn_params=CNN_PARAMS, gat_params=GAT_PARAMS,
            transformer_params=TRANSFORMER_PARAMS,
            pooling_params=POOLING_PARAMS, mlp_params=MLP_PARAMS
        ).to(device)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        if USE_COSINE_SCHEDULE:
            num_training_steps = len(train_loader) * NUM_EPOCHS
            num_warmup_steps = len(train_loader) * WARMUP_EPOCHS
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_model_state, history = train_fold(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            device, NUM_EPOCHS, PATIENCE, fold_num, SAVE_DIR,
            ACCUMULATION_STEPS, USE_AMP
        )
        
        model.load_state_dict(best_model_state)
        print(f"Testing Fold {fold_num} with session-level voting...")
        test_acc, conf_mat, true_labels, pred_labels = test_with_voting(model, test_loader, device, USE_AMP)
        
        print(f"FOLD {fold_num} RESULTS (Session-level voting)")
        print(f"Accuracy: {test_acc:.4f}")
        print(f"Confusion Matrix:\n{conf_mat}")
        print("Classification Report:")
        print(classification_report(true_labels, pred_labels, target_names=['twoBACK', 'Flanker', 'PVT']))
        
        fold_results.append({
            'fold': fold_num,
            'history': history,
            'test_accuracy': test_acc,
            'confusion_matrix': conf_mat.tolist(),
            'true_labels': [int(x) for x in true_labels],
            'pred_labels': [int(x) for x in pred_labels]
        })
        with open(os.path.join(SAVE_DIR, f'fold_{fold_num}_results.json'), 'w') as f:
            json.dump({k: v if not isinstance(v, np.ndarray) else v.tolist()
                      for k, v in fold_results[-1].items()}, f, indent=4)
        
        del model, train_loader, val_loader, test_loader, optimizer, scheduler, best_model_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    test_accs = [result['test_accuracy'] for result in fold_results]
    print(f"Test Accuracies per fold: {[f'{acc:.4f}' for acc in test_accs]}")
    print(f"Mean Test Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    total_conf_matrix = np.zeros((3, 3), dtype=int)
    for result in fold_results:
        total_conf_matrix += np.array(result['confusion_matrix'])
    print("Aggregated Confusion Matrix (across all folds):")
    print(total_conf_matrix)
    
    aggregate_results = {
        'mean_test_accuracy': float(np.mean(test_accs)),
        'std_test_accuracy': float(np.std(test_accs)),
        'min_test_accuracy': float(np.min(test_accs)),
        'max_test_accuracy': float(np.max(test_accs)),
        'fold_accuracies': test_accs,
        'aggregated_confusion_matrix': total_conf_matrix.tolist(),
        'configuration': {
            'num_nodes': NUM_NODES,
            'batch_size': BATCH_SIZE,
            'accumulation_steps': ACCUMULATION_STEPS,
            'effective_batch_size': BATCH_SIZE * ACCUMULATION_STEPS,
            'use_amp': USE_AMP,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'n_folds': N_FOLDS,
            'k_neighbors': K_NEIGHBORS,
            'window_size_ms': WINDOW_SIZE_MS,
            'step_size_ms': STEP_SIZE_MS,
            'validation_mode': 'session-level',
            'early_stopping_metric': 'validation_loss',
            'cnn_params': CNN_PARAMS,
            'gat_params': GAT_PARAMS,
            'transformer_params': TRANSFORMER_PARAMS,
            'pooling_params': POOLING_PARAMS,
            'mlp_params': MLP_PARAMS
        }
    }
    
    with open(os.path.join(SAVE_DIR, 'aggregate_results.json'), 'w') as f:
        json.dump(aggregate_results, f, indent=4)
    
    print(f"All results saved to: {SAVE_DIR}")
    print("Training complete!")


if __name__ == "__main__":
    main()