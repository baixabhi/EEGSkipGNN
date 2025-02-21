
!pip install torch-geometric

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Dataset
import numpy as np
from scipy.signal import butter, lfilter, welch
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch_geometric.loader import DataLoader

from torch_geometric.data import Data, Dataset
from scipy.signal import butter, lfilter

# DEAP_CHANNEL_LIST = [
#     'FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3',
#     'P7', 'PO3', 'O1', 'OZ', 'PZ', 'FP2', 'AF4', 'FZ', 'F4', 'F8', 'FC6', 'FC2',
#     'CZ', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
# ]

# DEAP_ADJACENCY_LIST = {
#     'FP1': ['AF3'], 'FP2': ['AF4'], 'AF3': ['FP1', 'FZ', 'F3'],
#     'AF4': ['FP2', 'F4', 'FZ'], 'F7': ['FC5'], 'F3': ['AF3', 'FC1', 'FC5'],
#     'FZ': ['AF3', 'AF4', 'FC2', 'FC1'], 'F4': ['AF4', 'FC6', 'FC2'],
#     'F8': ['FC6'], 'FC5': ['F7', 'F3', 'C3', 'T7'],
#     'FC1': ['F3', 'FZ', 'CZ', 'C3'], 'FC2': ['FZ', 'F4', 'C4', 'CZ'],
#     'FC6': ['F4', 'F8', 'T8', 'C4'], 'T7': ['FC5', 'CP5'],
#     'C3': ['FC5', 'FC1', 'CP1', 'CP5'], 'CZ': ['FC1', 'FC2', 'CP2', 'CP1'],
#     'C4': ['FC2', 'FC6', 'CP6', 'CP2'], 'T8': ['FC6', 'CP6'],
#     'CP5': ['T7', 'C3', 'P3', 'P7'], 'CP1': ['C3', 'CZ', 'PZ', 'P3'],
#     'CP2': ['CZ', 'C4', 'P4', 'PZ'], 'CP6': ['C4', 'T8', 'P8', 'P4'],
#     'P7': ['CP5'], 'P3': ['CP5', 'CP1', 'PO3'],
#     'PZ': ['CP1', 'CP2', 'PO4', 'PO3'], 'P4': ['CP2', 'CP6', 'PO4'],
#     'P8': ['CP6'], 'PO3': ['P3', 'PZ', 'OZ', 'O1'],
#     'PO4': ['PZ', 'P4', 'O2', 'OZ'], 'O1': ['PO3', 'OZ'],
#     'OZ': ['PO3', 'PO4', 'O2', 'O1'], 'O2': ['PO4', 'OZ']
# }

data_path = "/kaggle/input/deap-dataset/deap-dataset/data_preprocessed_python"
eeg_data = []
eeg_labels = []

for file in os.listdir(data_path):
    if file.endswith(".dat"):
        data = np.load(os.path.join(data_path, file), allow_pickle=True)
        eeg_data.append(data['data'][:, :32, :])
        eeg_labels.append(data['labels'])

eeg_data = np.array(eeg_data)
eeg_labels = np.array(eeg_labels)

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def normalize_eeg(data):
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    std[std == 0] = 1  # Avoid division by zero
    return (data - mean) / std

def segment_eeg(data, window_size, overlap, num_channels=64):
    num_participants, num_trials, _, num_samples = data.shape
    step = int(window_size * (1 - overlap))
    num_segments = (num_samples - window_size) // step + 1

    if num_segments <= 0:
        raise ValueError(
            f"Window size ({window_size}) and overlap ({overlap}) are incompatible "
            f"with the number of samples ({num_samples})."
        )

    segments = np.zeros((
        num_participants * num_trials * num_segments,
        num_channels,
        window_size
    ))

    segment_idx = 0
    for p in range(num_participants):
        for t in range(num_trials):
            for i in range(num_segments):
                start = i * step
                end = start + window_size
                segments[segment_idx] = data[p, t, :, start:end]
                segment_idx += 1

    return segments

def extract_frequency_bands(segment, fs, bands):
    freqs, psd = welch(segment, fs=fs, nperseg=fs)
    band_powers = []
    for low, high in bands.values():
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        if idx_band.any():
            band_power = np.trapz(psd[idx_band], freqs[idx_band])
        else:
            band_power = 0
        band_powers.append(band_power)
    return np.array(band_powers)

def process_eeg_data(eeg_data, eeg_labels, fs=128, window_size=128, overlap=0.5, num_channels=64):
    eeg_filtered = np.array([
        bandpass_filter(trial, 1, 50, fs)
        for trial in eeg_data.reshape(-1, eeg_data.shape[-1])
    ]).reshape(eeg_data.shape)

    eeg_normalized = normalize_eeg(eeg_filtered)
    eeg_segments = segment_eeg(eeg_normalized, window_size, overlap, num_channels)

    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }

    # Extract frequency band features
    eeg_band_features = np.array([
        [extract_frequency_bands(segment_channel, fs, bands) for segment_channel in segment]
        for segment in eeg_segments
    ])

    eeg_segments = np.concatenate((eeg_segments, eeg_band_features), axis=2)

    segments_per_trial = eeg_segments.shape[0] // (eeg_labels.shape[0] * eeg_labels.shape[1])
    labels = eeg_labels.reshape(-1, eeg_labels.shape[-1])
    segment_labels = np.repeat(labels, segments_per_trial, axis=0)

    return eeg_segments, segment_labels

def encode_labels(labels):
    valence = labels[:, 0] >= 5.0
    arousal = labels[:, 1] >= 5.0
    encoded = (valence.astype(int) * 2) + arousal.astype(int)
    return encoded

def create_eeg_graph_structure(num_channels=64, adjacency_list=DEAP_ADJACENCY_LIST):
    channel_to_index = {channel: idx for idx, channel in enumerate(DEAP_CHANNEL_LIST[:num_channels])}

    edges = []
    for channel, neighbors in adjacency_list.items():
        if channel in channel_to_index:
            src_idx = channel_to_index[channel]
            for neighbor in neighbors:
                if neighbor in channel_to_index:
                    dest_idx = channel_to_index[neighbor]
                    edges.append([src_idx, dest_idx])
                    edges.append([dest_idx, src_idx])  # Bidirectional
    edges = list(set(tuple(edge) for edge in edges))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index

class EEGGraphDataset(Dataset):
    def __init__(self, features, labels, num_channels=64):
        super().__init__()
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.num_channels = num_channels
        self.edge_index = create_eeg_graph_structure(num_channels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        return Data(x=x, edge_index=self.edge_index, y=self.labels[idx])

num_channels = 32
eeg_segments, segment_labels = process_eeg_data(eeg_data, eeg_labels, num_channels=num_channels)
four_class_labels = encode_labels(segment_labels)

def add_eeg_noise(data, noise_level=0.5):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def augment_minority_classes(features, labels, target_counts):

    """Augments minority classes using realistic EEG noise"""

    unique_labels, counts = np.unique(labels, return_counts=True)
    augmented_features = list(features)
    augmented_labels = list(labels)

    for label, count in zip(unique_labels, counts):
        if label in target_counts and count < target_counts[label]:
            num_to_augment = target_counts[label] - count
            indices = np.where(labels == label)[0]

            for _ in range(num_to_augment):
                idx = np.random.choice(indices)
                noisy_sample = add_eeg_noise(features[idx])
                augmented_features.append(noisy_sample)
                augmented_labels.append(label)

    return np.array(augmented_features), np.array(augmented_labels)

# Target counts
target_counts = {
    0: 80000,
    1: 50000,
    2: 50000,
    3: 50000
}

# Augment data
eeg_segments_augmented, four_class_labels_augmented = augment_minority_classes(
    eeg_segments, four_class_labels, target_counts
)


output_dir = "/kaggle/working/processed_data"

os.makedirs(output_dir, exist_ok=True)


np.save(os.path.join(output_dir, "eeg_segments_augmented.npy"), eeg_segments_augmented)


np.save(os.path.join(output_dir, "four_class_labels_augmented.npy"), four_class_labels_augmented)

print(f"Processed data saved to: {output_dir}")

import zipfile

zip_filename = "/kaggle/working/processed_data.zip"

with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            zipf.write(file_path, arcname=filename)

print(f"Zip file created: {zip_filename}")



!pip install torch-geometric

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data, Dataset
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from scipy.signal import butter, lfilter, welch
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch_geometric.loader import DataLoader
import os


# --- Data Loading ---
eeg_segments_augmented_path = "/kaggle/input/eeg-processed/eeg_segments_augmented.npy"
four_class_labels_augmented_path = "/kaggle/input/eeg-processed/four_class_labels_augmented.npy"

eeg_segments_augmented = np.load(eeg_segments_augmented_path)
four_class_labels_augmented = np.load(four_class_labels_augmented_path)


DEAP_CHANNEL_LIST = [
    'FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3',
    'P7', 'PO3', 'O1', 'OZ', 'PZ', 'FP2', 'AF4', 'FZ', 'F4', 'F8', 'FC6', 'FC2',
    'CZ', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
]

DEAP_ADJACENCY_LIST = {
    'FP1': ['AF3'], 'FP2': ['AF4'], 'AF3': ['FP1', 'FZ', 'F3'],
    'AF4': ['FP2', 'F4', 'FZ'], 'F7': ['FC5'], 'F3': ['AF3', 'FC1', 'FC5'],
    'FZ': ['AF3', 'AF4', 'FC2', 'FC1'], 'F4': ['AF4', 'FC6', 'FC2'],
    'F8': ['FC6'], 'FC5': ['F7', 'F3', 'C3', 'T7'],
    'FC1': ['F3', 'FZ', 'CZ', 'C3'], 'FC2': ['FZ', 'F4', 'C4', 'CZ'],
    'FC6': ['F4', 'F8', 'T8', 'C4'], 'T7': ['FC5', 'CP5'],
    'C3': ['FC5', 'FC1', 'CP1', 'CP5'], 'CZ': ['FC1', 'FC2', 'CP2', 'CP1'],
    'C4': ['FC2', 'FC6', 'CP6', 'CP2'], 'T8': ['FC6', 'CP6'],
    'CP5': ['T7', 'C3', 'P3', 'P7'], 'CP1': ['C3', 'CZ', 'PZ', 'P3'],
    'CP2': ['CZ', 'C4', 'P4', 'PZ'], 'CP6': ['C4', 'T8', 'P8', 'P4'],
    'P7': ['CP5'], 'P3': ['CP5', 'CP1', 'PO3'],
    'PZ': ['CP1', 'CP2', 'PO4', 'PO3'], 'P4': ['CP2', 'CP6', 'PO4'],
    'P8': ['CP6'], 'PO3': ['P3', 'PZ', 'OZ', 'O1'],
    'PO4': ['PZ', 'P4', 'O2', 'OZ'], 'O1': ['PO3', 'OZ'],
    'OZ': ['PO3', 'PO4', 'O2', 'O1'], 'O2': ['PO4', 'OZ']
}

def create_eeg_graph_structure(num_channels=64, adjacency_list=DEAP_ADJACENCY_LIST):
    channel_to_index = {channel: idx for idx, channel in enumerate(DEAP_CHANNEL_LIST[:num_channels])}

    edges = []
    for channel, neighbors in adjacency_list.items():
        if channel in channel_to_index:
            src_idx = channel_to_index[channel]
            for neighbor in neighbors:
                if neighbor in channel_to_index:
                    dest_idx = channel_to_index[neighbor]
                    edges.append([src_idx, dest_idx])
                    edges.append([dest_idx, src_idx])  # Bidirectional
    edges = list(set(tuple(edge) for edge in edges))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index

class EEGGraphDataset(Dataset):
    def __init__(self, features, labels, num_channels=64):
        super().__init__()
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.num_channels = num_channels
        self.edge_index = create_eeg_graph_structure(num_channels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        return Data(x=x, edge_index=self.edge_index, y=self.labels[idx])



# --- Multi-Head Attention ---
class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.query_conv = nn.Conv1d(in_channels, num_heads * head_dim, kernel_size=1, bias=False)
        self.key_conv = nn.Conv1d(in_channels, num_heads * head_dim, kernel_size=1, bias=False)
        self.value_conv = nn.Conv1d(in_channels, num_heads * head_dim, kernel_size=1, bias=False)
        self.output_conv = nn.Conv1d(num_heads * head_dim, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, time_steps = x.size()

        # Linear projections
        queries = self.query_conv(x).view(batch_size, self.num_heads, self.head_dim, time_steps)
        keys = self.key_conv(x).view(batch_size, self.num_heads, self.head_dim, time_steps)
        values = self.value_conv(x).view(batch_size, self.num_heads, self.head_dim, time_steps)

        # Scaled dot-product attention
        attention_scores = torch.einsum("bhdt,bhde->bhte", queries, keys) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.einsum("bhte,bhde->bhdt", attention_weights, values)

        # Concatenate heads and project
        concatenated = attended_values.reshape(batch_size, channels, time_steps)
        output = self.output_conv(concatenated)

        return output

class TemporalSpectralModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.num_channels = num_channels

        # Initial projection to match channels
        self.input_proj = nn.Conv1d(num_channels, out_channels, 1)

        # Calculate the output channels for each temporal conv branch
        self.branch_channels = out_channels // len(kernel_sizes)

        # Temporal Convolution branches with different kernel sizes
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(out_channels, self.branch_channels,
                     kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])

        # Frequency attention
        self.freq_attention = nn.Sequential(
            nn.Conv1d(out_channels, out_channels // 2, 1),
            nn.BatchNorm1d(out_channels // 2),
            nn.ELU(),
            nn.Conv1d(out_channels // 2, out_channels, 1),
            nn.Sigmoid()
        )

        # Cross-channel relationship module
        self.channel_mixer = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
            nn.LayerNorm(out_channels * 2),
            nn.ELU(),
            nn.Linear(out_channels * 2, out_channels),
            nn.Sigmoid()
        )

        # Output projection - note the input channels is now correctly calculated
        total_channels = (self.branch_channels * len(kernel_sizes)) + out_channels
        self.output_proj = nn.Sequential(
            nn.Conv1d(total_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ELU()
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Initial projection
        x = self.input_proj(x)  # [batch_size, out_channels, time_steps]

        # Multi-scale temporal convolutions
        temporal_outputs = []
        for conv in self.temporal_convs:
            temporal_outputs.append(conv(x))
        temporal_features = torch.cat(temporal_outputs, dim=1)  # [batch_size, branch_channels * num_branches, time_steps]

        # Frequency domain analysis
        fft_features = torch.fft.rfft(x, dim=2)
        magnitude = torch.abs(fft_features)
        phase = torch.angle(fft_features)
        freq_features = torch.fft.irfft(magnitude * torch.exp(1j * phase), n=x.size(2), dim=2)

        # Apply frequency attention
        freq_weights = self.freq_attention(freq_features)
        weighted_freq = freq_features * freq_weights

        # Combine features
        combined = torch.cat([temporal_features, weighted_freq], dim=1)
        output = self.output_proj(combined)

        return output

class SkipEEGNet(nn.Module):
    def __init__(self, in_channels, time_steps, hidden_channels=64, num_classes=4, num_channels=64):
        super().__init__()
        self.num_channels = num_channels
        self.time_steps = time_steps

        # Initial feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(time_steps, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU()
        )

        # Multiple GCN layers with skip connections
        self.conv_layers = nn.ModuleList([
            GCNConv(hidden_channels, hidden_channels) for _ in range(3)
        ])

        # Batch normalization layers
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(3)
        ])

        # Transition layers for skip connections
        self.transition_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels * (i + 1), hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ELU()
            ) for i in range(3)
        ])

        # Final GCN layer
        self.final_conv = GCNConv(hidden_channels, hidden_channels * 2)
        self.final_bn = nn.BatchNorm1d(hidden_channels * 2)

        # Global attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1)
        )

        # Modified classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Initial feature projection
        x = self.feature_proj(x)

        # Store intermediate features for skip connections
        intermediate_features = []
        current_x = x

        # GCN layers with skip connections
        for i, (conv, bn, transition) in enumerate(zip(self.conv_layers, self.bn_layers, self.transition_layers)):
            # Apply GCN layer
            conv_out = F.elu(bn(conv(current_x, edge_index)))
            intermediate_features.append(conv_out)

            # Concatenate all previous features
            all_features = torch.cat([feat for feat in intermediate_features], dim=-1)

            # Apply transition to maintain channel dimensions
            current_x = transition(all_features)
            current_x = F.dropout(current_x, p=0.5, training=self.training)

        # Final GCN layer
        x = F.elu(self.final_bn(self.final_conv(current_x, edge_index)))

        # Apply attention mechanism
        # First, we need to get the node features for each graph in the batch
        x_dense, mask = to_dense_batch(x, batch)  # [batch_size, max_nodes, features]

        # Calculate attention scores
        scores = self.attention(x_dense)  # [batch_size, max_nodes, 1]

        # Mask out padding nodes
        mask = mask.unsqueeze(-1).float()  # [batch_size, max_nodes, 1]
        scores = scores.masked_fill(~mask.bool(), float('-inf'))

        # Apply softmax over the nodes dimension
        attention_weights = F.softmax(scores, dim=1)  # [batch_size, max_nodes, 1]

        # Apply attention weights
        x_weighted = x_dense * attention_weights  # [batch_size, max_nodes, features]

        # Sum over nodes to get graph-level representation
        x = x_weighted.sum(dim=1)  # [batch_size, features]

        # Apply classifier
        out = self.classifier(x)

        return out

def train_evaluate_deep_gnn(features, labels, batch_size=32, num_epochs=50, learning_rate=0.0003, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = EEGGraphDataset(X_train, y_train)
    val_dataset = EEGGraphDataset(X_val, y_val)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    time_steps = features.shape[2]
    model = SkipEEGNet(
        in_channels=time_steps,
        time_steps=time_steps,
        num_channels=64
    ).to(device)

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_accuracy = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            outputs = model(batch)
            loss = loss_fn(outputs, batch.y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(batch.y).sum().item()
            train_total += batch.y.size(0)

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                loss = loss_fn(outputs, batch.y)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(batch.y).sum().item()
                val_total += batch.y.size(0)

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total

        # Update learning rate
        scheduler.step(val_loss)

        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        print("----------------------------------------")

    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\nBest Validation Accuracy: {best_val_accuracy:.4f}")

    return model


model = train_evaluate_deep_gnn(
    eeg_segments_augmented,
    four_class_labels_augmented,
    batch_size=32,
    num_epochs=50,
    learning_rate=0.0003,
    patience=10
)





