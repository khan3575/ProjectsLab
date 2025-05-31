import numpy as np
import torch
from torch_geometric.data import Data
from skimage.segmentation import slic, felzenszwalb
from skimage.measure import regionprops
from sklearn.neighbors import NearestNeighbors
import cv2
from scipy import ndimage

class ImageToGraphConverter:
    """Convert medical images to graph representation"""
    
    def __init__(self, method='superpixel', n_segments=200, k_neighbors=8):
        self.method = method
        self.n_segments = n_segments
        self.k_neighbors = k_neighbors
        
    
    def convert(self, multi_modal_image, segmentation_mask):
        """
        Convert multi-modal medical image to graph
        
        Args:
            multi_modal_image: numpy array of shape (C, H, W) where C is number of modalities
            segmentation_mask: numpy array of shape (H, W) with ground truth labels
            
        Returns:
            torch_geometric.data.Data object
        """
        # Ensure correct input format
        if multi_modal_image.ndim == 3:
            channels, height, width = multi_modal_image.shape
        else:
            raise ValueError(f"Expected 3D input (C, H, W), got shape {multi_modal_image.shape}")
        
        # Ensure segmentation mask is 2D
        if segmentation_mask.ndim != 2:
            raise ValueError(f"Expected 2D segmentation mask, got shape {segmentation_mask.shape}")
        
        if segmentation_mask.shape != (height, width):
            raise ValueError(f"Segmentation mask shape {segmentation_mask.shape} doesn't match image shape {(height, width)}")
        
        # Create superpixels using the first modality (T1)
        base_image = multi_modal_image[0]  # Use T1 modality for superpixel generation
        
        # Normalize for superpixel generation
        if base_image.max() > 0:
            normalized_base = ((base_image - base_image.min()) / (base_image.max() - base_image.min()) * 255).astype(np.uint8)
        else:
            normalized_base = np.zeros_like(base_image, dtype=np.uint8)
        
        # Generate superpixels - FIX: explicitly set channel_axis=None for 2D grayscale
        if self.method == 'superpixel':
            segments = slic(
                normalized_base, 
                n_segments=self.n_segments,
                compactness=10,
                sigma=1,
                start_label=0,
                channel_axis=None  # FIX: explicitly set for 2D grayscale
            )
        elif self.method == 'felzenszwalb':
            segments = felzenszwalb(
                normalized_base,
                scale=100,
                sigma=0.5,
                min_size=50
            )
        else:
            raise ValueError(f"Unknown segmentation method: {self.method}")
        
        # Extract node features and labels
        node_features, node_labels, centroids = self._extract_node_features(
            multi_modal_image, segmentation_mask, segments
        )
        
        # Build graph edges
        edge_index = self._build_edges(centroids, segments)
        
        # Convert to PyTorch tensors
        x = torch.FloatTensor(node_features)
        y = torch.LongTensor(node_labels)
        edge_index = torch.LongTensor(edge_index)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        
        # Add metadata
        data.num_nodes = len(node_features)
        data.image_shape = (height, width)
        data.num_modalities = channels
        
        return data
    
    def _extract_node_features(self, multi_modal_image, segmentation_mask, segments):
        """Extract features for each superpixel (node)"""
        channels, height, width = multi_modal_image.shape
        
        # Get unique segment IDs
        unique_segments = np.unique(segments)
        num_nodes = len(unique_segments)
        
        # Initialize feature matrix
        # Features: [mean_intensity_per_modality, std_intensity_per_modality, area]
        num_features = channels * 2 + 1  # mean + std for each modality + area
        node_features = np.zeros((num_nodes, num_features))
        node_labels = np.zeros(num_nodes, dtype=np.int64)
        centroids = np.zeros((num_nodes, 2))
        
        for i, segment_id in enumerate(unique_segments):
            # Create mask for current segment
            segment_mask = (segments == segment_id)
            
            # Skip if segment is empty
            if not np.any(segment_mask):
                continue
            
            # Extract features for each modality
            feature_idx = 0
            
            for c in range(channels):
                modality_data = multi_modal_image[c]
                segment_pixels = modality_data[segment_mask]
                
                # Mean intensity
                node_features[i, feature_idx] = np.mean(segment_pixels) if len(segment_pixels) > 0 else 0
                feature_idx += 1
                
                # Standard deviation
                node_features[i, feature_idx] = np.std(segment_pixels) if len(segment_pixels) > 0 else 0
                feature_idx += 1
            
            # Area (normalized by total image size)
            area = np.sum(segment_mask) / (height * width)
            node_features[i, feature_idx] = area
            
            # Centroid
            y_coords, x_coords = np.where(segment_mask)
            if len(y_coords) > 0:
                centroids[i, 0] = np.mean(y_coords)
                centroids[i, 1] = np.mean(x_coords)
            
            # Label (majority vote within segment)
            segment_labels = segmentation_mask[segment_mask]
            if len(segment_labels) > 0:
                unique_labels, counts = np.unique(segment_labels, return_counts=True)
                node_labels[i] = unique_labels[np.argmax(counts)]
            else:
                node_labels[i] = 0  # Background
        
        return node_features, node_labels, centroids
    
    def _build_edges(self, centroids, segments):
        """Build graph edges based on spatial proximity"""
        num_nodes = len(centroids)
        
        if num_nodes <= 1:
            return np.array([[], []], dtype=np.int64)
        
        # Use k-nearest neighbors based on centroid distances
        k = min(self.k_neighbors, num_nodes - 1)
        
        # Fit k-NN model
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(centroids)
        distances, indices = nbrs.kneighbors(centroids)
        
        # Build edge list
        edge_list = []
        
        for i in range(num_nodes):
            for j in range(1, len(indices[i])):  # Skip self (index 0)
                neighbor_idx = indices[i][j]
                
                # Add bidirectional edges
                edge_list.append([i, neighbor_idx])
                edge_list.append([neighbor_idx, i])
        
        # Remove duplicates and convert to numpy array
        edge_list = list(set(map(tuple, edge_list)))
        
        if len(edge_list) == 0:
            return np.array([[], []], dtype=np.int64)
        
        edge_index = np.array(edge_list).T
        
        return edge_index

def test_converter():
    """Test the graph converter with synthetic data"""
    print("Testing ImageToGraphConverter...")
    
    # Create synthetic multi-modal image (4 modalities)
    height, width = 128, 128
    channels = 4
    
    # Simulate T1, T1ce, T2, FLAIR
    multi_modal = np.random.rand(channels, height, width).astype(np.float32)
    
    # Create synthetic segmentation with 4 classes
    segmentation = np.zeros((height, width), dtype=np.uint8)
    segmentation[40:80, 40:80] = 1  # Tumor core
    segmentation[30:90, 30:90] = 2  # Edema (larger region)
    segmentation[35:85, 35:85] = 3  # Enhancing tumor
    
    # Create converter
    converter = ImageToGraphConverter(method='superpixel', n_segments=100, k_neighbors=6)
    
    # Convert to graph
    try:
        graph_data = converter.convert(multi_modal, segmentation)
        
        print(f"✅ Conversion successful!")
        print(f"  Nodes: {graph_data.x.shape[0]}")
        print(f"  Features per node: {graph_data.x.shape[1]}")
        print(f"  Edges: {graph_data.edge_index.shape[1]}")
        print(f"  Labels: {torch.unique(graph_data.y)}")
        print(f"  Image shape: {graph_data.image_shape}")
        print(f"  Modalities: {graph_data.num_modalities}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_converter()