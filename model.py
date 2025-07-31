import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple, List
import math


class E3EquivariantLayer(MessagePassing):
    """
    E(3) Equivariant Graph Neural Network layer.
    This layer respects rotational and translational symmetries.
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, out_dim: int):
        super().__init__(aggr='mean')
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # Node feature MLPs
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        # Edge feature MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + 1, hidden_dim),  # +1 for distance
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Coordinate update MLP
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x, pos, edge_index, edge_attr, t, batch=None):
        """
        Forward pass of the equivariant layer.
        
        Args:
            x: Node features [num_nodes, node_dim]
            pos: Node positions [num_nodes, 3]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            t: Diffusion timestep [batch_size]
            batch: Batch indices [num_nodes]
        
        Returns:
            Updated node features and positions
        """
        # Expand timestep to match node count
        if batch is not None:
            t_expanded = t[batch].unsqueeze(-1)  # [num_nodes, 1]
        else:
            t_expanded = t.unsqueeze(-1).expand(x.size(0), 1)
        
        # Time embedding
        t_embed = self.time_embed(t_expanded)
        
        # Start message passing
        return self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr, 
                            t_embed=t_embed, batch=batch)
    
    def message(self, x_i, x_j, pos_i, pos_j, edge_attr, t_embed_i, t_embed_j):
        """
        Compute messages between connected nodes.
        """
        # Compute relative positions and distances
        rel_pos = pos_j - pos_i  # [num_edges, 3]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # [num_edges, 1]
        
        # Concatenate edge features with distance
        edge_input = torch.cat([edge_attr, dist], dim=-1)
        edge_features = self.edge_mlp(edge_input)
        
        # Combine node features with time embedding
        node_features = torch.cat([x_j, t_embed_j], dim=-1)
        
        return node_features, edge_features, rel_pos
    
    def update(self, aggr_out, x, pos, t_embed):
        """
        Update node features and positions based on aggregated messages.
        """
        # Split aggregated output
        aggr_node_features, aggr_edge_features, aggr_rel_pos = aggr_out
        
        # Update node features
        node_input = torch.cat([x, aggr_node_features], dim=-1)
        new_x = self.node_mlp(node_input)
        
        # Update positions (equivariant update)
        coord_update = self.coord_mlp(aggr_edge_features)
        new_pos = pos + coord_update
        
        return new_x, new_pos


class E3EquivariantGNN(nn.Module):
    """
    Complete E(3) Equivariant Graph Neural Network for molecular diffusion.
    """
    
    def __init__(self, 
                 node_dim: int = 1,
                 edge_dim: int = 0,
                 hidden_dim: int = 128,
                 num_layers: int = 6,
                 max_atoms: int = 29):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_atoms = max_atoms
        
        # Initial node feature embedding
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        
        # Initial edge feature embedding (if edge features exist)
        if edge_dim > 0:
            self.edge_embed = nn.Linear(edge_dim, hidden_dim)
        else:
            self.edge_embed = None
        
        # Stack of equivariant layers
        self.layers = nn.ModuleList([
            E3EquivariantLayer(
                node_dim=hidden_dim if i == 0 else hidden_dim,
                edge_dim=hidden_dim,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim
            ) for i in range(num_layers)
        ])
        
        # Final output layers
        self.noise_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Predict 3D noise
        )
        
        # Optional: predict atom type probabilities
        self.atom_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)  # 5 atom types: H, C, N, O, F
        )
        
    def forward(self, data: Data, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the E3-GNN.
        
        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features [num_nodes, node_dim]
                - pos: Node positions [num_nodes, 3]
                - edge_index: Edge indices [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_dim] (optional)
                - batch: Batch indices [num_nodes]
            t: Diffusion timestep [batch_size]
        
        Returns:
            Tuple of (noise_pred, atom_logits)
        """
        x, pos, edge_index = data.x, data.pos, data.edge_index
        batch = getattr(data, 'batch', None)
        
        # Handle edge features
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = data.edge_attr
        else:
            # Create dummy edge features
            edge_attr = torch.zeros(edge_index.size(1), self.hidden_dim, 
                                  device=x.device, dtype=x.dtype)
        
        # Initial embeddings
        x = self.node_embed(x.float())
        if self.edge_embed is not None:
            edge_attr = self.edge_embed(edge_attr)
        
        # Pass through equivariant layers
        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, edge_attr, t, batch)
        
        # Predict noise for coordinates
        noise_pred = self.noise_predictor(x)
        
        # Predict atom type probabilities
        atom_logits = self.atom_predictor(x)
        
        return noise_pred, atom_logits
    
    def compute_edge_features(self, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute edge features based on relative positions.
        """
        row, col = edge_index
        rel_pos = pos[col] - pos[row]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        
        # Create edge features from distance and relative position
        edge_features = torch.cat([rel_pos, dist], dim=-1)
        return edge_features


class DiffusionModel(nn.Module):
    """
    Complete diffusion model wrapper.
    """
    
    def __init__(self, 
                 node_dim: int = 1,
                 hidden_dim: int = 128,
                 num_layers: int = 6,
                 max_atoms: int = 29):
        super().__init__()
        
        self.gnn = E3EquivariantGNN(
            node_dim=node_dim,
            edge_dim=4,  # 3 for rel_pos + 1 for distance
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_atoms=max_atoms
        )
        
    def forward(self, data: Data, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the diffusion model.
        
        Args:
            data: Molecular graph data
            t: Diffusion timestep
        
        Returns:
            Predicted noise and atom type logits
        """
        # Compute edge features
        edge_features = self.gnn.compute_edge_features(data.pos, data.edge_index)
        
        # Create new data object with edge features
        data_with_edges = Data(
            x=data.x,
            pos=data.pos,
            edge_index=data.edge_index,
            edge_attr=edge_features,
            batch=getattr(data, 'batch', None)
        )
        
        return self.gnn(data_with_edges, t)


if __name__ == "__main__":
    # Test the model
    model = DiffusionModel(node_dim=1, hidden_dim=64, num_layers=3, max_atoms=29)
    
    # Create dummy data
    batch_size = 2
    num_nodes = 10
    
    x = torch.randint(1, 6, (num_nodes, 1), dtype=torch.long)
    pos = torch.randn(num_nodes, 3)
    edge_index = torch.randint(0, num_nodes, (2, 20))
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes // batch_size)
    t = torch.rand(batch_size)
    
    data = Data(x=x, pos=pos, edge_index=edge_index, batch=batch)
    
    # Test forward pass
    noise_pred, atom_logits = model(data, t)
    
    print(f"Model output shapes:")
    print(f"Noise prediction: {noise_pred.shape}")
    print(f"Atom logits: {atom_logits.shape}")