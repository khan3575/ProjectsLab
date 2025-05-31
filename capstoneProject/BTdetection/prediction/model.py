import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch

class GNNTumorSegmentation(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim=64,
                 num_classes=4,  # Background, Necrosis, Edema, Enhancing
                 num_layers=3,
                 gnn_type='GCN',
                 dropout=0.2,
                 use_attention=True):
        super(GNNTumorSegmentation, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Input projection: projects input features into hidden space.
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Define GNN layers and corresponding normalization layers.
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            if gnn_type == 'GCN':
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'GAT':
                # For GAT we use multiple heads if not on the final layer.
                heads = 8 if i < num_layers - 1 else 1
                # If not final layer, adjust out_dim accordingly.
                out_dim = hidden_dim // heads if i < num_layers - 1 else hidden_dim
                self.gnn_layers.append(GATConv(hidden_dim, out_dim, heads=heads, concat=(i < num_layers - 1)))
            elif gnn_type == 'SAGE':
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"Unsupported gnn_type: {gnn_type}")
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Optionally, apply an attention mechanism on node features.
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        
        # Classification: process output from GNN layers for segmentation.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Edge feature processing (if available). Always active since condition is True.
        self.edge_proj = nn.Linear(1, 1)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = getattr(data, 'batch', None)
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Process edge attributes if available.
        edge_weight = None
        if edge_attr is not None and self.edge_proj is not None:
            edge_weight = self.edge_proj(edge_attr).squeeze(-1)
            # Ensure edge_weight has the correct size.
            if edge_weight.size(0) != edge_index.size(1):
                print(f"Warning: edge_weight size {edge_weight.size()} doesn't match edge_index size {edge_index.size()}")
                edge_weight = None
        
        # Passing through GNN layers.
        for i, (gnn_layer, batch_norm) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            residual = x
            # Apply the corresponding GNN layer.
            if self.gnn_type == 'GAT':
                x = gnn_layer(x, edge_index)
            elif self.gnn_type == 'SAGE':
                x = gnn_layer(x, edge_index)
            else:  # Default to GCN
                if edge_weight is not None and edge_weight.size(0) == edge_index.size(1):
                    x = gnn_layer(x, edge_index, edge_weight=edge_weight)
                else:
                    x = gnn_layer(x, edge_index)
            
            # Use BatchNorm if batch information is available; otherwise, use layer normalization.
            if batch is not None:
                x = batch_norm(x)
            else:
                x = F.layer_norm(x, [x.shape[-1]])
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection: add previous representation if shapes match.
            if i > 0 and residual.shape == x.shape:
                x = x + residual
        
        # Apply attention mechanism if enabled.
        attention_weights = None
        if self.use_attention:
            attention_weights = self.attention(x)
            x = x * attention_weights
        
        # Apply classifier to get final output.
        out = self.classifier(x)
        
        return out, attention_weights

class MultiScaleGNN(nn.Module):
    """
    Multi-scale GNN for hierarchical tumor segmentation.
    It uses multiple GNNTumorSegmentation models at different scales
    and fuses their graph-level representations.
    """
    
    def __init__(self, 
                 input_dim,
                 hidden_dim=64,
                 num_classes=4,
                 scales=[100, 200, 400],  # Different superpixel scales.
                 gnn_type='GCN'):
        super(MultiScaleGNN, self).__init__()
        
        self.scales = scales
        self.num_scales = len(scales)
        
        # Create individual GNNs for each scale.
        self.gnns = nn.ModuleList()
        for _ in scales:
            self.gnns.append(GNNTumorSegmentation(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=hidden_dim,  # Output features, not final segmentation classes.
                num_layers=2,
                gnn_type=gnn_type
            ))
        
        # Fusion layer to combine multi-scale features.
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * self.num_scales, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, multi_scale_data):
        """
        Args:
            multi_scale_data: List of Data objects for different scales.
        """
        scale_features = []
        
        for i, data in enumerate(multi_scale_data):
            features, _ = self.gnns[i](data)
            # Global pooling to get a graph-level representation.
            if hasattr(data, 'batch') and data.batch is not None:
                pooled_features = global_mean_pool(features, data.batch)
            else:
                pooled_features = features.mean(dim=0, keepdim=True)
            scale_features.append(pooled_features)
        
        # Concatenate multi-scale features.
        fused_features = torch.cat(scale_features, dim=-1)
        # Final classification.
        output = self.fusion(fused_features)
        
        return output

# Loss Functions

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=1, gamma=2, num_classes=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        # Compute cross-entropy loss without reduction.
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Replace any NaNs with zeros.
        focal_loss = torch.where(torch.isnan(focal_loss), torch.zeros_like(focal_loss), focal_loss)
        return focal_loss.mean()

class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""
    
    def __init__(self, smooth=1, num_classes=4):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        # Convert to probabilities safely.
        inputs = F.softmax(inputs, dim=1)
        
        # Flatten predictions and targets.
        inputs_flat = inputs.view(-1, self.num_classes)
        targets_flat = targets.view(-1).long()
        
        # Clamp target range.
        targets_flat = torch.clamp(targets_flat, 0, self.num_classes - 1)
        # One-hot encode targets.
        targets_one_hot = F.one_hot(targets_flat, num_classes=self.num_classes).float()
        
        # Calculate Dice score per class.
        dice_scores = []
        for i in range(self.num_classes):
            input_class = inputs_flat[:, i]
            target_class = targets_one_hot[:, i]
            intersection = (input_class * target_class).sum()
            union = input_class.sum() + target_class.sum()
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Return Dice loss.
        dice_loss = 1 - torch.stack(dice_scores).mean()
        return torch.clamp(dice_loss, 0, 1)

class CombinedLoss(nn.Module):
    """Combined Focal + Dice Loss."""
    
    def __init__(self, focal_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        # If any loss component is NaN, replace with zero.
        if torch.isnan(focal):
            focal = torch.tensor(0.0, device=inputs.device)
        if torch.isnan(dice):
            dice = torch.tensor(0.0, device=inputs.device)
        
        return self.focal_weight * focal + self.dice_weight * dice

def create_test_data(num_nodes=200, input_dim=9, num_classes=4):
    """Create properly formatted test data for the GNN."""
    
    # Create node features.
    x = torch.randn(num_nodes, input_dim)
    
    # Create edges.
    edge_list = []
    for i in range(num_nodes):
        # Connect each node to up to 5 random neighbors.
        num_neighbors = min(5, num_nodes - 1)
        neighbors = torch.randperm(num_nodes)[:num_neighbors]
        for neighbor in neighbors:
            if neighbor != i:  # Avoid self-connections.
                edge_list.append([i, neighbor.item()])
    
    # Convert edge list to tensor.
    if edge_list:
        edges = torch.tensor(edge_list).T
        # Make the graph undirected.
        edge_index = torch.cat([edges, edges.flip(0)], dim=1)
        # Remove duplicate entries.
        edge_index = torch.unique(edge_index, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Create edge attributes.
    num_edges = edge_index.size(1)
    edge_attr = torch.randn(num_edges, 1) if num_edges > 0 else None
    
    # Create labels for each node.
    y = torch.randint(0, num_classes, (num_nodes,))
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# Test the model
if __name__ == "__main__":
    print("Testing GNN Model...")
    
    # Create sample data.
    num_nodes = 200
    input_dim = 9  # e.g., 4 MRI modalities + 5 geometric features.
    num_classes = 4
    
    data = create_test_data(num_nodes, input_dim, num_classes)
    
    print(f"Created test data:")
    print(f"  Nodes: {data.x.shape[0]}")
    print(f"  Node features: {data.x.shape[1]}")
    print(f"  Edges: {data.edge_index.shape[1]}")
    print(f"  Edge attributes: {data.edge_attr.shape if data.edge_attr is not None else 'None'}")
    
    # Test basic GNN model.
    print("\nTesting basic GNNTumorSegmentation...")
    model = GNNTumorSegmentation(input_dim=input_dim, hidden_dim=64, num_classes=num_classes)
    with torch.no_grad():
        output, attention = model(data)
    
    print(f"Input shape: {data.x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attention.shape if attention is not None else 'None'}")
    
    # Test loss functions.
    print("\nTesting loss functions...")
    criterion = CombinedLoss()
    loss = criterion(output, data.y)
    print(f"Combined loss: {loss.item():.4f}")
    
    # Test different GNN types.
    for gnn_type in ['GCN', 'GAT', 'SAGE']:
        print(f"\nTesting {gnn_type} type...")
        model = GNNTumorSegmentation(input_dim=input_dim, gnn_type=gnn_type)
        with torch.no_grad():
            output, _ = model(data)
        print(f"{gnn_type} output shape: {output.shape}")
    
    # Test with CUDA if available.
    if torch.cuda.is_available():
        print(f"\nTesting on GPU...")
        device = torch.device('cuda')
        model = GNNTumorSegmentation(input_dim=input_dim).to(device)
        data = data.to(device)
        with torch.no_grad():
            output, _ = model(data)
        print(f"GPU output shape: {output.shape}")
        print(f"GPU memory used: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
    
    print("\nGNN model test completed successfully!")