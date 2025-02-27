from helper_functions.halfedge import *

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Dataset, DataLoader

from torch_geometric.nn import HeteroConv, SAGEConv, GCNConv
from torch_geometric.data import Batch, HeteroData

from torch_scatter import scatter_mean
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR, CosineAnnealingLR, OneCycleLR

def set_seed(seed=42):
    """
    Set the random seed for reproducibility across all relevant libraries.
    
    Args:
        seed (int): Seed value for random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to normalize features
def normalize_features(tensor):
    return (tensor - tensor.mean(dim=0)) / (tensor.std(dim=0) + 1e-8)
    
# Function to split data_list into train and test
def split_train_test(df, remove_indices=None):
    """
    Split data_list into training and test sets based on 'train' or 'test' in the file path.

    Args:
        data_list (list): List of tuples (file_path, label).

    Returns:
        tuple: (train_list, test_list)
    """

    df = df.drop(remove_indices)

    class_to_int = {cls: idx for idx, cls in enumerate(df['class'].unique())}
    data_list = [(row['path'], class_to_int[row['class']]) for _, row in df.iterrows()]

    train_list = [item for item in data_list if 'train' in item[0].lower()]
    test_list = [item for item in data_list if 'test' in item[0].lower()]
    return train_list, test_list

class MeshDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        off_file, label = self.data_list[idx]
        mesh = load_half_edge_structure(off_file)
        vertices = mesh['vertices']
        half_edges = mesh['half_edges']
        faces = mesh['faces']

        edge_dict = {}
        edge_features_list = []
        edge_index = 0
        for he_idx, he in half_edges.items():
            v1 = int(he['origin'])
            next_he = half_edges[he['next']]
            v2 = int(next_he['origin'])
            edge_key = tuple(sorted([v1, v2]))
            if edge_key not in edge_dict:
                edge_dict[edge_key] = edge_index
                edge_features_list.append(float(he['length']))
                edge_features_list.append(1.0)
                edge_index += 1

        vertex_features = torch.tensor(
            [[float(c) for c in v['coords']] + [float(n) for n in v['normal']] + [1.0] for v in vertices.values()],
            dtype=torch.float
        )
        edge_features = torch.tensor(edge_features_list, dtype=torch.float).view(-1, 2)
        face_features = torch.tensor(
            [[float(f['area'])] + [float(n) for n in f['normal']] + [1.0] for f in faces.values()],
            dtype=torch.float
        )

        vertex_features = normalize_features(vertex_features)
        edge_features = normalize_features(edge_features)
        face_features = normalize_features(face_features)

        vv_edge_index = torch.tensor([[v1, v2] for v1, v2 in edge_dict.keys()], dtype=torch.long).t()
        ve_edge_index = torch.tensor([[v, e] for (v1, v2), e in edge_dict.items() for v in [v1, v2]], dtype=torch.long).t()
        vf_edge_index = torch.tensor(
            [[int(v_idx), f_idx] for f_idx, face in enumerate(faces.values()) for v_idx in face['vertices']],
            dtype=torch.long
        ).t()

        def _get_face_half_edges(face, half_edges):
            he_idx = face['half_edge']
            start_he = he_idx
            he_indices = []
            while True:
                he_indices.append(he_idx)
                he_idx = half_edges[he_idx]['next']
                if he_idx == start_he:
                    break
            return he_indices

        ef_edge_index = torch.tensor(
            [[e_idx, f_idx] for f_idx, face in enumerate(faces.values()) 
             for he_idx in _get_face_half_edges(face, half_edges) 
             for e_idx in [edge_dict[tuple(sorted([int(half_edges[he_idx]['origin']), int(half_edges[half_edges[he_idx]['next']]['origin'])]))]]],
            dtype=torch.long
        ).t()

        data = HeteroData()
        data['v'].x = vertex_features
        data['e'].x = edge_features
        data['f'].x = face_features
        data['v', 'to', 'v'].edge_index = vv_edge_index
        data['v', 'to', 'e'].edge_index = ve_edge_index
        data['v', 'to', 'f'].edge_index = vf_edge_index
        data['e', 'to', 'f'].edge_index = ef_edge_index
        data['e', 'to', 'v'].edge_index = ve_edge_index.flip(0)
        data['f', 'to', 'v'].edge_index = vf_edge_index.flip(0)
        data['f', 'to', 'e'].edge_index = ef_edge_index.flip(0)
        data.y = torch.tensor([label], dtype=torch.long)
        return data

class MeshDataset_WOCORDS(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        off_file, label = self.data_list[idx]
        mesh = load_half_edge_structure(off_file)
        vertices = mesh['vertices']
        half_edges = mesh['half_edges']
        faces = mesh['faces']

        edge_dict = {}
        edge_features_list = []
        edge_index = 0
        for he_idx, he in half_edges.items():
            v1 = int(he['origin'])
            next_he = half_edges[he['next']]
            v2 = int(next_he['origin'])
            edge_key = tuple(sorted([v1, v2]))
            if edge_key not in edge_dict:
                edge_dict[edge_key] = edge_index
                edge_features_list.append(float(he['length']))
                edge_features_list.append(1.0)
                edge_index += 1

        vertex_features = torch.tensor(
            [[float(c) for c in v['coords']] + [1.0] for v in vertices.values()],
            dtype=torch.float
        )
        edge_features = torch.tensor(edge_features_list, dtype=torch.float).view(-1, 2)
        face_features = torch.tensor(
            [[float(f['area'])] + [float(n) for n in f['normal']] + [1.0] for f in faces.values()],
            dtype=torch.float
        )

        vertex_features = normalize_features(vertex_features)
        edge_features = normalize_features(edge_features)
        face_features = normalize_features(face_features)

        vv_edge_index = torch.tensor([[v1, v2] for v1, v2 in edge_dict.keys()], dtype=torch.long).t()
        ve_edge_index = torch.tensor([[v, e] for (v1, v2), e in edge_dict.items() for v in [v1, v2]], dtype=torch.long).t()
        vf_edge_index = torch.tensor(
            [[int(v_idx), f_idx] for f_idx, face in enumerate(faces.values()) for v_idx in face['vertices']],
            dtype=torch.long
        ).t()

        def _get_face_half_edges(face, half_edges):
            he_idx = face['half_edge']
            start_he = he_idx
            he_indices = []
            while True:
                he_indices.append(he_idx)
                he_idx = half_edges[he_idx]['next']
                if he_idx == start_he:
                    break
            return he_indices

        ef_edge_index = torch.tensor(
            [[e_idx, f_idx] for f_idx, face in enumerate(faces.values()) 
             for he_idx in _get_face_half_edges(face, half_edges) 
             for e_idx in [edge_dict[tuple(sorted([int(half_edges[he_idx]['origin']), int(half_edges[half_edges[he_idx]['next']]['origin'])]))]]],
            dtype=torch.long
        ).t()

        data = HeteroData()
        data['v'].x = vertex_features
        data['e'].x = edge_features
        data['f'].x = face_features
        data['v', 'to', 'v'].edge_index = vv_edge_index
        data['v', 'to', 'e'].edge_index = ve_edge_index
        data['v', 'to', 'f'].edge_index = vf_edge_index
        data['e', 'to', 'f'].edge_index = ef_edge_index
        data['e', 'to', 'v'].edge_index = ve_edge_index.flip(0)
        data['f', 'to', 'v'].edge_index = vf_edge_index.flip(0)
        data['f', 'to', 'e'].edge_index = ef_edge_index.flip(0)
        data.y = torch.tensor([label], dtype=torch.long)
        return data

class SNNFirstLayer(nn.Module):
    def __init__(self, hidden_dim=128, features=[7, 2, 5]):
        super(SNNFirstLayer, self).__init__()
        self.conv = HeteroConv({
            ('v', 'to', 'v'): SAGEConv(features[0], hidden_dim),
            ('v', 'to', 'e'): SAGEConv((features[0], features[1]), hidden_dim),
            ('v', 'to', 'f'): SAGEConv((features[0], features[2]), hidden_dim),
            ('e', 'to', 'v'): SAGEConv((features[1], features[0]), hidden_dim),
            ('e', 'to', 'f'): SAGEConv((features[1], features[2]), hidden_dim),
            ('f', 'to', 'v'): SAGEConv((features[2], features[0]), hidden_dim),
            ('f', 'to', 'e'): SAGEConv((features[2], features[1]), hidden_dim),
        }, aggr='mean')
        self.self_v = nn.Linear(features[0], hidden_dim)
        self.self_e = nn.Linear(features[1], hidden_dim)
        self.self_f = nn.Linear(features[2], hidden_dim)

    def forward(self, data):
        x_dict = {
            'v': data['v'].x,
            'e': data['e'].x,
            'f': data['f'].x
        }
        edge_index_dict = {
            ('v', 'to', 'v'): data['v', 'to', 'v'].edge_index,
            ('v', 'to', 'e'): data['v', 'to', 'e'].edge_index,
            ('v', 'to', 'f'): data['v', 'to', 'f'].edge_index,
            ('e', 'to', 'v'): data['e', 'to', 'v'].edge_index,
            ('e', 'to', 'f'): data['e', 'to', 'f'].edge_index,
            ('f', 'to', 'v'): data['f', 'to', 'v'].edge_index,
            ('f', 'to', 'e'): data['f', 'to', 'e'].edge_index,
        }
        conv_out = self.conv(x_dict, edge_index_dict)
        x_v = F.relu(self.self_v(x_dict['v']) + conv_out['v'])
        x_e = F.relu(self.self_e(x_dict['e']) + conv_out['e'])
        x_f = F.relu(self.self_f(x_dict['f']) + conv_out['f'])
        out = HeteroData()
        out['v'].x = x_v
        out['e'].x = x_e
        out['f'].x = x_f
        # Preserve batch attributes
        if hasattr(data['v'], 'batch'):
            out['v'].batch = data['v'].batch
        if hasattr(data['e'], 'batch'):
            out['e'].batch = data['e'].batch
        if hasattr(data['f'], 'batch'):
            out['f'].batch = data['f'].batch
        out['v', 'to', 'v'].edge_index = edge_index_dict[('v', 'to', 'v')]
        out['v', 'to', 'e'].edge_index = edge_index_dict[('v', 'to', 'e')]
        out['v', 'to', 'f'].edge_index = edge_index_dict[('v', 'to', 'f')]
        out['e', 'to', 'v'].edge_index = edge_index_dict[('e', 'to', 'v')]
        out['e', 'to', 'f'].edge_index = edge_index_dict[('e', 'to', 'f')]
        out['f', 'to', 'v'].edge_index = edge_index_dict[('f', 'to', 'v')]
        out['f', 'to', 'e'].edge_index = edge_index_dict[('f', 'to', 'e')]
        out.y = data.y
        out.num_graphs = getattr(data, 'num_graphs', 1)  # Preserve num_graphs
        return out

class SNNHiddenLayer(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SNNHiddenLayer, self).__init__()
        self.conv = HeteroConv({
            ('v', 'to', 'v'): SAGEConv(hidden_dim, hidden_dim),
            ('v', 'to', 'e'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ('v', 'to', 'f'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ('e', 'to', 'v'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ('e', 'to', 'f'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ('f', 'to', 'v'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ('f', 'to', 'e'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        }, aggr='mean')
        self.self_v = nn.Linear(hidden_dim, hidden_dim)
        self.self_e = nn.Linear(hidden_dim, hidden_dim)
        self.self_f = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        x_dict = {
            'v': data['v'].x,
            'e': data['e'].x,
            'f': data['f'].x
        }
        edge_index_dict = {
            ('v', 'to', 'v'): data['v', 'to', 'v'].edge_index,
            ('v', 'to', 'e'): data['v', 'to', 'e'].edge_index,
            ('v', 'to', 'f'): data['v', 'to', 'f'].edge_index,
            ('e', 'to', 'v'): data['e', 'to', 'v'].edge_index,
            ('e', 'to', 'f'): data['e', 'to', 'f'].edge_index,
            ('f', 'to', 'v'): data['f', 'to', 'v'].edge_index,
            ('f', 'to', 'e'): data['f', 'to', 'e'].edge_index,
        }
        conv_out = self.conv(x_dict, edge_index_dict)
        x_v = F.relu(self.self_v(x_dict['v']) + conv_out['v'])
        x_e = F.relu(self.self_e(x_dict['e']) + conv_out['e'])
        x_f = F.relu(self.self_f(x_dict['f']) + conv_out['f'])
        out = HeteroData()
        out['v'].x = x_v
        out['e'].x = x_e
        out['f'].x = x_f
        # Preserve batch attributes
        if hasattr(data['v'], 'batch'):
            out['v'].batch = data['v'].batch
        if hasattr(data['e'], 'batch'):
            out['e'].batch = data['e'].batch
        if hasattr(data['f'], 'batch'):
            out['f'].batch = data['f'].batch
        out['v', 'to', 'v'].edge_index = edge_index_dict[('v', 'to', 'v')]
        out['v', 'to', 'e'].edge_index = edge_index_dict[('v', 'to', 'e')]
        out['v', 'to', 'f'].edge_index = edge_index_dict[('v', 'to', 'f')]
        out['e', 'to', 'v'].edge_index = edge_index_dict[('e', 'to', 'v')]
        out['e', 'to', 'f'].edge_index = edge_index_dict[('e', 'to', 'f')]
        out['f', 'to', 'v'].edge_index = edge_index_dict[('f', 'to', 'v')]
        out['f', 'to', 'e'].edge_index = edge_index_dict[('f', 'to', 'e')]
        out.y = data.y
        out.num_graphs = getattr(data, 'num_graphs', 1)  # Preserve num_graphs
        return out

# Updated SNN with corrected batch access
class SNN(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=10, num_layers=2):
        super(SNN, self).__init__()
        self.first_layer = SNNFirstLayer(hidden_dim)
        self.hidden_layers = nn.ModuleList([SNNHiddenLayer(hidden_dim) for _ in range(num_layers - 1)])
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def _initialize_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, data):
        data = self.first_layer(data)
        for layer in self.hidden_layers:
            data = layer(data)
        if hasattr(data, 'num_graphs') and data.num_graphs > 1:
            v_batch = data['v'].batch  # Direct access
            e_batch = data['e'].batch
            f_batch = data['f'].batch
            num_graphs = data.num_graphs
            v_pooled = scatter_mean(data['v'].x, v_batch, dim=0, dim_size=num_graphs)
            e_pooled = scatter_mean(data['e'].x, e_batch, dim=0, dim_size=num_graphs)
            f_pooled = scatter_mean(data['f'].x, f_batch, dim=0, dim_size=num_graphs)
            x = torch.cat([v_pooled, e_pooled, f_pooled], dim=1)
        else:
            v_pooled = data['v'].x.mean(dim=0)
            e_pooled = data['e'].x.mean(dim=0)
            f_pooled = data['f'].x.mean(dim=0)
            x = torch.cat([v_pooled, e_pooled, f_pooled], dim=0)
        out = self.fc(x)
        return out 

# Enhanced version of SNN with regularization techniques
class EnhancedSNN(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=10, num_layers=2, 
                 dropout_rate=0.2, use_batch_norm=True, use_weight_norm=False, 
                 residual_connections=True, conv_type='sage', features=[7, 2, 5]):
        """
        Enhanced SNN model with various regularization techniques.
        
        Args:
            hidden_dim (int): Size of hidden dimension
            num_classes (int): Number of output classes
            num_layers (int): Number of hidden layers
            dropout_rate (float): Dropout probability
            use_batch_norm (bool): Whether to use batch normalization
            use_weight_norm (bool): Whether to use weight normalization
            residual_connections (bool): Whether to use residual connections
            conv_type (str): Type of convolution ('sage' or 'gcn')
            features (list): Feature dimensions for vertices, edges, and faces
        """
        super(EnhancedSNN, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.residual_connections = residual_connections
        self.num_layers = num_layers
        self.features = features
        
        # Choose convolution type
        self.Conv = SAGEConv if conv_type.lower() == 'sage' else GCNConv
        
        # First layer
        self.first_layer = self._create_first_layer(hidden_dim, features)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(self._create_hidden_layer(hidden_dim))
        
        # Batch normalization layers
        if use_batch_norm:
            self.v_bn = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
            self.e_bn = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
            self.f_bn = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        
        # Output layers with dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim * 3, 256)
        self.fc_bn = nn.BatchNorm1d(256) if use_batch_norm else nn.Identity()
        self.fc2 = nn.Linear(256, num_classes)
        
        # Apply weight normalization if requested
        if use_weight_norm:
            self.fc1 = nn.utils.weight_norm(self.fc1)
            self.fc2 = nn.utils.weight_norm(self.fc2)
        
        self._initialize_weights()
    
    def _create_first_layer(self, hidden_dim, features):
        return HeteroConv({
            ('v', 'to', 'v'): self.Conv(features[0], hidden_dim),
            ('v', 'to', 'e'): self.Conv((features[0], features[1]), hidden_dim),
            ('v', 'to', 'f'): self.Conv((features[0], features[2]), hidden_dim),
            ('e', 'to', 'v'): self.Conv((features[1], features[0]), hidden_dim),
            ('e', 'to', 'f'): self.Conv((features[1], features[2]), hidden_dim),
            ('f', 'to', 'v'): self.Conv((features[2], features[0]), hidden_dim),
            ('f', 'to', 'e'): self.Conv((features[2], features[1]), hidden_dim),
        }, aggr='mean')
    
    def _create_hidden_layer(self, hidden_dim):
        return HeteroConv({
            ('v', 'to', 'v'): self.Conv(hidden_dim, hidden_dim),
            ('v', 'to', 'e'): self.Conv((hidden_dim, hidden_dim), hidden_dim),
            ('v', 'to', 'f'): self.Conv((hidden_dim, hidden_dim), hidden_dim),
            ('e', 'to', 'v'): self.Conv((hidden_dim, hidden_dim), hidden_dim),
            ('e', 'to', 'f'): self.Conv((hidden_dim, hidden_dim), hidden_dim),
            ('f', 'to', 'v'): self.Conv((hidden_dim, hidden_dim), hidden_dim),
            ('f', 'to', 'e'): self.Conv((hidden_dim, hidden_dim), hidden_dim),
        }, aggr='mean')
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _forward_layer(self, data, layer_idx, x_dict, edge_index_dict, original_x=None):
        if layer_idx == 0:
            conv_out = self.first_layer(x_dict, edge_index_dict)
        else:
            conv_out = self.hidden_layers[layer_idx-1](x_dict, edge_index_dict)
        
        # Apply residual connections if specified and not the first layer
        if self.residual_connections and original_x is not None:
            x_v = conv_out['v'] + original_x['v']
            x_e = conv_out['e'] + original_x['e']
            x_f = conv_out['f'] + original_x['f']
        else:
            x_v = conv_out['v']
            x_e = conv_out['e']
            x_f = conv_out['f']
        
        # Apply batch normalization if specified
        if self.use_batch_norm:
            x_v = self.v_bn[layer_idx](x_v)
            x_e = self.e_bn[layer_idx](x_e)
            x_f = self.f_bn[layer_idx](x_f)
        
        # Apply ReLU and dropout
        x_v = F.relu(x_v)
        x_e = F.relu(x_e)
        x_f = F.relu(x_f)
        
        x_v = F.dropout(x_v, p=self.dropout_rate, training=self.training)
        x_e = F.dropout(x_e, p=self.dropout_rate, training=self.training)
        x_f = F.dropout(x_f, p=self.dropout_rate, training=self.training)
        
        return {'v': x_v, 'e': x_e, 'f': x_f}
    
    def forward(self, data):
        x_dict = {
            'v': data['v'].x,
            'e': data['e'].x,
            'f': data['f'].x
        }
        
        edge_index_dict = {
            ('v', 'to', 'v'): data['v', 'to', 'v'].edge_index,
            ('v', 'to', 'e'): data['v', 'to', 'e'].edge_index,
            ('v', 'to', 'f'): data['v', 'to', 'f'].edge_index,
            ('e', 'to', 'v'): data['e', 'to', 'v'].edge_index,
            ('e', 'to', 'f'): data['e', 'to', 'f'].edge_index,
            ('f', 'to', 'v'): data['f', 'to', 'v'].edge_index,
            ('f', 'to', 'e'): data['f', 'to', 'e'].edge_index,
        }
        
        # Process through layers
        for i in range(self.num_layers):
            original_x = x_dict.copy() if i > 0 and self.residual_connections else None
            x_dict = self._forward_layer(data, i, x_dict, edge_index_dict, original_x)
        
        # Pooling
        if hasattr(data, 'num_graphs') and data.num_graphs > 1:
            v_batch = data['v'].batch
            e_batch = data['e'].batch
            f_batch = data['f'].batch
            num_graphs = data.num_graphs
            v_pooled = scatter_mean(x_dict['v'], v_batch, dim=0, dim_size=num_graphs)
            e_pooled = scatter_mean(x_dict['e'], e_batch, dim=0, dim_size=num_graphs)
            f_pooled = scatter_mean(x_dict['f'], f_batch, dim=0, dim_size=num_graphs)
            x = torch.cat([v_pooled, e_pooled, f_pooled], dim=1)
        else:
            v_pooled = x_dict['v'].mean(dim=0)
            e_pooled = x_dict['e'].mean(dim=0)
            f_pooled = x_dict['f'].mean(dim=0)
            x = torch.cat([v_pooled, e_pooled, f_pooled], dim=0)
        
        # Final classification layers
        x = self.dropout(x)
        x = F.relu(self.fc_bn(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

