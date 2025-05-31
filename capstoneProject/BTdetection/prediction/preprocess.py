import os
import sys
import pickle
import glob
import numpy as np
import nibabel as nib
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# Set up paths: adjust these if your directory structure differs.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming this file is placed inside the "data" folder and that the dataset is in "brats-2021-task1":
data_root_default = os.path.join(current_dir, "brats-2021-task1")

# Add parent directory to sys.path so that we can import modules from the preprocessing folder.
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from graph_construction import ImageToGraphConverter  # Make sure this module is available

class BraTSProcessor:
    """
    Processes the BraTS-2021 dataset and converts MRI slices to graph representations.
    
    Uses superpixel segmentation to build the graph structures and provides functionality
    for processing the full dataset as well as creating statistics.
    """

    def __init__(self, data_root: str = data_root_default, output_dir: str = None):
        self.data_root = data_root
        # If output_dir is not provided, create a 'processed' folder inside data_root.
        self.output_dir = output_dir or os.path.join(self.data_root, 'processed')
        self.graph_converter = ImageToGraphConverter(
            method='superpixel',
            n_segments=500,   # More segments for brain MRI
            k_neighbors=12
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def get_patient_folders(self) -> list:
        """
        Retrieves and returns all patient folders from the training data.
        Patient folders should start with 'BraTS2021_'.
        """
        training_path = os.path.join(self.data_root, 'raw', 'BraTS2021_Training_Data')
        if not os.path.exists(training_path):
            raise FileNotFoundError(f"Training data not found at: {training_path}")

        patient_folders = [
            os.path.join(training_path, folder)
            for folder in os.listdir(training_path)
            if os.path.isdir(os.path.join(training_path, folder)) and folder.startswith('BraTS2021_')
        ]
        print(f"Found {len(patient_folders)} patient folders in {training_path}")
        return sorted(patient_folders)

    def load_patient_data(self, patient_folder: str):
        """
        Loads the four MRI modalities and the segmentation for a single patient.
        Returns a tuple (data, segmentation, patient_id) or None if any modality is missing.
        """
        patient_id = os.path.basename(patient_folder)
        modalities = {
            't1': f'{patient_id}_t1.nii.gz',
            't1ce': f'{patient_id}_t1ce.nii.gz', 
            't2': f'{patient_id}_t2.nii.gz',
            'flair': f'{patient_id}_flair.nii.gz'
        }
        seg_file = f'{patient_id}_seg.nii.gz'
        data = {}

        for modality, filename in modalities.items():
            filepath = os.path.join(patient_folder, filename)
            if os.path.exists(filepath):
                nii = nib.load(filepath)
                data[modality] = nii.get_fdata().astype(np.float32)
            else:
                print(f"Warning: {filepath} not found")
                return None

        seg_path = os.path.join(patient_folder, seg_file)
        if os.path.exists(seg_path):
            seg_nii = nib.load(seg_path)
            segmentation = seg_nii.get_fdata().astype(np.uint8)
        else:
            print(f"Warning: {seg_path} not found")
            return None

        return data, segmentation, patient_id

    def preprocess_volume(self, volume: np.ndarray, percentile_clip: float = 99.5) -> np.ndarray:
        """
        Preprocesses the MRI volume by clipping extreme values and normalizing.
        """
        volume = volume.copy()
        non_zero = volume > 0
        if np.any(non_zero):
            upper_bound = np.percentile(volume[non_zero], percentile_clip)
            volume = np.clip(volume, 0, upper_bound)
            volume = volume / (upper_bound + 1e-8)  # Normalize to [0, 1] with eps to avoid division by zero
        return volume

    def extract_brain_slices(self, data: dict, segmentation: np.ndarray, brain_thresh: int = 1000) -> list:
        """
        Extracts 2D slices that have significant brain content.
        Returns a list of slices along with modality and segmentation information.
        """
        t1 = data['t1']
        h, w, d = t1.shape
        valid_slices = []

        for slice_idx in range(d):
            t1_slice = data['t1'][:, :, slice_idx]
            t1ce_slice = data['t1ce'][:, :, slice_idx]
            t2_slice = data['t2'][:, :, slice_idx]
            flair_slice = data['flair'][:, :, slice_idx]
            seg_slice = segmentation[:, :, slice_idx]

            brain_voxels = np.sum(t1_slice > 0)
            tumor_voxels = np.sum(seg_slice > 0)

            if brain_voxels > brain_thresh:
                valid_slices.append({
                    'slice_idx': slice_idx,
                    't1': t1_slice,
                    't1ce': t1ce_slice,
                    't2': t2_slice,
                    'flair': flair_slice,
                    'segmentation': seg_slice,
                    'brain_voxels': brain_voxels,
                    'tumor_voxels': tumor_voxels
                })

        print(f"Found {len(valid_slices)} valid slices out of {d} total slices")
        return valid_slices

    def process_slice_to_graph(self, slice_data: dict):
        """
        Converts a single 2D slice into a graph using the provided graph converter.
        Returns the graph data object or None if conversion fails.
        """
        # Stack modalities to create a multi-channel image (channels, height, width)
        multi_modal = np.stack([
            slice_data['t1'],
            slice_data['t1ce'], 
            slice_data['t2'],
            slice_data['flair']
        ], axis=0)

        # Preprocess each modality channel
        for i in range(4):
            multi_modal[i] = self.preprocess_volume(multi_modal[i])

        segmentation = slice_data['segmentation']
        if segmentation.ndim != 2:
            segmentation = segmentation.squeeze()

        if multi_modal.shape[1:] != segmentation.shape:
            print(f"Shape mismatch: image {multi_modal.shape[1:]} vs mask {segmentation.shape}")
            return None

        try:
            graph_data = self.graph_converter.convert(multi_modal, segmentation)
            return graph_data
        except Exception as e:
            print(f"Error converting slice to graph: {e}")
            print(f"  Multi-modal shape: {multi_modal.shape}")
            print(f"  Segmentation shape: {segmentation.shape}")
            return None

    def process_patient(self, patient_folder: str, max_slices: int = 20) -> list:
        """
        Processes a single patient folder, converting valid slices into graphs.
        Returns a list of graph objects with added metadata.
        """
        print(f"Processing patient: {os.path.basename(patient_folder)}")
        result = self.load_patient_data(patient_folder)
        if result is None:
            return []

        data, segmentation, patient_id = result
        valid_slices = self.extract_brain_slices(data, segmentation)
        if not valid_slices:
            print(f"No valid slices found for {patient_id}")
            return []

        # Sort slices by tumor content (highest first) and select up to max_slices
        valid_slices.sort(key=lambda x: x['tumor_voxels'], reverse=True)
        selected_slices = valid_slices[:max_slices]

        graphs = []
        for idx, slice_data in enumerate(selected_slices):
            graph = self.process_slice_to_graph(slice_data)
            if graph is not None:
                # Add metadata directly into the graph object as new attributes.
                graph.patient_id = patient_id
                graph.slice_idx = slice_data['slice_idx']
                graph.tumor_voxels = slice_data['tumor_voxels']
                graphs.append(graph)
                if (idx + 1) % 5 == 0:
                    print(f"  Processed slice {idx+1}/{len(selected_slices)}")
        print(f"Converted {len(graphs)} slices to graphs for patient {patient_id}")
        return graphs

    def process_patients(self, patient_folders: list, max_slices_per_patient: int = 20) -> list:
        """
        Processes a list of patient folders.
        """
        all_graphs = []
        for folder in tqdm(patient_folders, desc="Processing patients"):
            try:
                patient_graphs = self.process_patient(folder, max_slices_per_patient)
                all_graphs.extend(patient_graphs)
            except Exception as e:
                print(f"Error processing {folder}: {e}")
                continue
        return all_graphs

    def save_graphs(self, graphs: list, filename: str):
        """
        Saves the list of graphs to a pickle file.
        """
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(graphs, f)
        print(f"Saved {len(graphs)} graphs to {filepath}")

    def save_statistics(self, graphs: list):
        """
        Computes and saves basic statistics about the processed graphs.
        """
        if not graphs:
            return

        stats = {
            'total_graphs': len(graphs),
            'patients': len(set(graph.patient_id for graph in graphs if hasattr(graph, 'patient_id'))),
            'avg_nodes': np.mean([graph.x.shape[0] for graph in graphs if hasattr(graph, 'x')]),
            'avg_edges': np.mean([graph.edge_index.shape[1] for graph in graphs if hasattr(graph, 'edge_index')]),
            'class_distribution': {}
        }
        try:
            all_labels = torch.cat([graph.y for graph in graphs if hasattr(graph, 'y')])
            unique_labels, counts = torch.unique(all_labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                stats['class_distribution'][f'class_{label.item()}'] = count.item()
        except Exception as e:
            print(f"Warning: Could not compute class distribution, {e}")

        stats_path = os.path.join(self.output_dir, 'dataset_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)

        print("\nDataset Statistics:")
        print(f"Total graphs: {stats['total_graphs']}")
        print(f"Patients: {stats['patients']}")
        print(f"Avg nodes per graph: {stats['avg_nodes']:.1f}")
        print(f"Avg edges per graph: {stats['avg_edges']:.1f}")
        print(f"Class distribution: {stats['class_distribution']}")

    def process_dataset(self, max_patients: int = None, max_slices_per_patient: int = 12):
        """
        Loads processed graphs if already available; otherwise, processes the full BraTS dataset.
        
        Arguments:
            max_patients (int, optional): Limit processing to the first max_patients patient folders.
            max_slices_per_patient (int): Maximum number of slices to process per patient.
        
        Returns:
            graphs (list): List of graph representations generated from the BraTS dataset.
        """
        graphs_file = os.path.join(self.output_dir, "brats_graphs.pkl")
        if os.path.exists(graphs_file):
            with open(graphs_file, 'rb') as f:
                graphs = pickle.load(f)
            print(f"Loaded processed graphs from {graphs_file}")
            return graphs
        else:
            print("Processed graphs not found.")
            print("Reprocessing the BraTS dataset to generate graph representations...")
            patient_folders = self.get_patient_folders()
            if max_patients is not None:
                patient_folders = patient_folders[:max_patients]
            graphs = self.process_patients(patient_folders, max_slices_per_patient)
            self.save_graphs(graphs, "brats_graphs.pkl")
            self.save_statistics(graphs)
            return graphs


if __name__ == "__main__":
    # For testing/debugging: process the dataset and output statistics.
    processor = BraTSProcessor()
    _ = processor.process_dataset(max_slices_per_patient=12)